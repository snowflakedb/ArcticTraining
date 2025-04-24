# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any
from typing import Union

import torch

from arctic_training.checkpoint.ds_engine import DSCheckpointEngine
from arctic_training.checkpoint.hf_engine import HFCheckpointEngine
from arctic_training.data.sft_factory import SFTDataFactory
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.model.liger_factory import LigerModelFactory
from arctic_training.optimizer.adam_factory import CPUAdamOptimizerFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.scheduler.hf_factory import HFSchedulerFactory
from arctic_training.tokenizer.hf_factory import HFTokenizerFactory
from arctic_training.trainer.trainer import Trainer
from arctic_training.trainer.utils import to_device

# XXX: this will be moved to deepspeed
if 1:
    from arctic_training.deepspeed import ChunkedMemEfficientLoss


class SFTTrainer(Trainer):
    name = "sft"
    data_factory: SFTDataFactory
    model_factory: Union[HFModelFactory, LigerModelFactory]
    checkpoint_engine: Union[DSCheckpointEngine, HFCheckpointEngine]
    optimizer_factory: Union[FusedAdamOptimizerFactory, CPUAdamOptimizerFactory]
    scheduler_factory: Union[HFSchedulerFactory]
    tokenizer_factory: Union[HFTokenizerFactory]

    def loss(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)


        enable_tiling = False

        if enable_tiling:
            model_with_head = self.model_unwrapped
            outputs = model_with_head.model(**batch, use_cache=False)
            hidden_states = outputs.last_hidden_state

            # logits_to_keep = 0
            # # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            #logits = model_with_head.lm_head(hidden_states[:, slice_indices, :])
            logits = model_with_head.lm_head(hidden_states)

            labels = None # XXX: will generalize later
            shift_labels = batch.pop("shift_labels") # XXX: will generalize later
            loss = model_with_head.loss_function(logits=logits, labels=labels, vocab_size=model_with_head.config.vocab_size, shift_labels=shift_labels)

            return loss




        if self.config.sequence_parallel_size == 1:
            outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
        else:

            # letting liger do fused logits+loss calculation
            # XXX: update to the latest liger version and it can handle shift_labels
            if 0:
                # This needs to be removed once liger new version is out track https://github.com/linkedin/Liger-Kernel/issues/675
                batch["labels"] = batch.pop("shift_labels")
                #batch["labels"] = None
                outputs = self.model(**batch, use_cache=False)
                # XXX: try to adapt this code to perform a faster and simpler activation checkpoint offloading instead of monkey patching torch's checkpoint mechanism
                # watch this Issue:
                # if experimenting with this disable activation_checkpointing_cpu_offload_threshold code branch
                #from arctic_training.monkey_patches import OffloadActivations
                #with OffloadActivations():
                #    outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                return loss

            # Ulysses SP
            # expectations:
            # 1. batch has labels replaced with shift_labels (which are already preshifted)
            # 2. this rank deals with a seqlen dimension shard so once the loss is calculated it needs to do a differentiable weighted loss average to get the grads right

            if "labels" in batch:
                raise ValueError(
                    "found labels in batch - they shouldn't be there, instead shift_labels should be there - check"
                    " that UlyssesAttentionHFDataLoaderWrapper has been applied to the original DataLoader object"
                )
            if "shift_labels" not in batch:
                raise ValueError(
                    "shift_labels are missing from the batch - check that UlyssesAttentionHFDataLoaderWrapper has been"
                    " applied to the original DataLoader object"
                )

            shift_labels = batch.pop("shift_labels")
            outputs = self.model(**batch, use_cache=False)
            logits = outputs.logits

            # XXX: parameterize
            num_loss_logit_shards: Any = "auto"
            #num_loss_logit_shards: Any = 2

            if all((shift_labels == -100).squeeze()):
                # this is the case where all labels in a micro-batch are -100 (very common for SFT) - CE returns `nan` in this case, so we don't want to call loss and instead create a differentiable loss `0` which will also set all the grads to `0` in `backward` - the effect of this is akin to a perfect score where the model needs no adjustment since grads will be all zeros.
                # XXX: should this be float and not the original dtype?
                loss = (logits.sum() * 0.0).float()
            else:
                if num_loss_logit_shards == "auto":
                    # parameterize to about 1GB fp32 logits shards
                    slice_size_in_gb = 1  # XXX: make configurable?
                    size_in_gb = logits.numel() * 4 / 2**30  # fp32
                    # the sp shard's seqlen sp shard can be easily not divisible by the derived number of chunked loss shards, so we use the uppper ceiling and allow the last chunk to be shorter than the rest
                    num_loss_logit_shards = math.ceil(size_in_gb / slice_size_in_gb)
                    # print(f"derived {num_loss_logit_shards} shards for size {size_in_gb}GB")
                if num_loss_logit_shards > 1:
                    # if shards == 1 this will lead to a higher memory usage then calling the normal loss function, so don't do that.

                    kwargs_to_shard = dict(
                        logits=logits,
                        shift_labels=shift_labels,
                    )
                    kwargs_to_pass = dict(
                        vocab_size=self.model_unwrapped.config.vocab_size
                    )
                    grad_requiring_tensor_key="logits"
                    seqlen = shift_labels.shape[1]

                    # since -100s shift_labels are ignored we have to perform a weighted average on each loss slice as each slice may contribute a different number of non- -100 labels
                    def loss_fn(logits=None, labels=None, shift_labels=None, vocab_size=0):
                        """
                        will return loss*good_items so that weighted loss could be calculated at the end
                        """
                        if all((shift_labels == -100).squeeze()):
                            # fake loss calculation, since CE will return nan, but grads will be set
                            # a normal loss_fn upcasts logits to float so match it
                            loss_sum = (logits.sum() * 0.0).float()
                        else:
                            good_items = sum((shift_labels != -100).squeeze())
                            loss = self.model_unwrapped.loss_function(
                                logits=logits,
                                labels=None,
                                vocab_size=vocab_size,
                                shift_labels=shift_labels,
                            )
                            loss_sum = loss*good_items
                        return loss_sum

                    from arctic_training.deepspeed import sequence_tiling_compute
                    total_loss_sum = sequence_tiling_compute(
                        loss_fn,
                        seqlen,
                        num_loss_logit_shards,
                        kwargs_to_shard,
                        kwargs_to_pass,
                        grad_requiring_tensor_key, # XXX: multiple keys for the general case?
                        reduction="sum",
                    )
                    total_good_items = sum((shift_labels != -100).squeeze())
                    loss = total_loss_sum / total_good_items

                    # # if shards == 1 this will lead to a higher memory usage then calling the normal loss function, so don't do that.
                    # loss = ChunkedMemEfficientLoss.apply(
                    #     self.model_unwrapped.loss_function,
                    #     logits,
                    #     self.model_unwrapped.config.vocab_size,
                    #     shift_labels,
                    #     num_loss_logit_shards,
                    # )
                else:
                    loss = self.model_unwrapped.loss_function(
                        logits=logits,
                        labels=None,
                        vocab_size=self.model_unwrapped.config.vocab_size,
                        shift_labels=shift_labels,
                    )

            # differentiable weighted per-shard-loss aggregation across ranks
            import torch.distributed.nn.functional

            losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=self.sp_group)
            good_tokens = sum((shift_labels != -100).view(-1))
            good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=self.sp_group)
            loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(self.sp_world_size)) / sum(
                good_tokens_per_rank
            )

        return loss
