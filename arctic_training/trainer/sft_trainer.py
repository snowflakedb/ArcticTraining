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
from typing import Union

import torch
import torch.distributed.nn.functional
from deepspeed.runtime.sequence_parallel.ulysses_sp import sequence_tiled_compute

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

        if self.config.sequence_parallel_size == 1:
            # if model.type=liger is configured - this will use a much more efficient fused
            # logits+loss liger kernel - using significantly less gpu memory and a bit faster
            # compute (liger fused logits+loss kernel does not repeat forward during backward)
            outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
            return loss

        # Ulysses SP expectations:
        # 1. batch has `labels`` replaced with `shift_labels`` (which are already preshifted in
        #    DataLoader)
        # 2. this rank deals with a seqlen dimension shard so once the loss is calculated it needs
        #    to do a differentiable weighted loss average to get the grads right

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

        shift_labels = batch["shift_labels"]

        # We have 2 implementation of efficient tiled logits+loss computation.
        # 1. Liger fused cross-entropy is the fastest/most memory efficient way - liger-kernel
        #    doesn't recompute forward inside backward, instead it computes the gradients in the
        #    forward path.
        # 2. But liger kernel isn't implemented for all HF Transformers models, so then we fall
        #    back onto our tiled logits+loss compute implementation that is almost as efficient
        #    memory-wise, but which has more compute overhead before backward re-runs forward. The
        #    total memory usage is very similar, but cuda cache flushes earlier if pushing close to
        #    OOM than liger.
        if self.config.model.type == "liger":
            # letting liger do fused logits+loss calculation
            outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss

        else:
            # Currently relying on an automatic num_shards derivation based on the goal that it'll
            # take approximately 1GB of fp32 logits in a shard, could make this configurable if
            # desired later. Less than 1GB doesn't seem to make much of an impact, but perhaps a
            # higher number will be more efficient as it'll run less shards.
            num_shards = "auto"
            if num_shards == "auto":
                # parameterize to about 1GB fp32 logits shards
                slice_size_in_gb = 1  # XXX: make configurable?
                bs, seqlen = shift_labels.shape
                vocab_size = self.model_unwrapped.config.vocab_size
                logits_numel = bs * seqlen * vocab_size
                size_in_gb = logits_numel * 4 / 2**30  # fp32
                # the sp shard's seqlen sp shard can be easily not divisible by the derived number
                # of chunked loss shards, so we use the uppper ceiling and allow the last chunk to
                # be shorter than the rest
                num_shards = math.ceil(size_in_gb / slice_size_in_gb)
                # print(f"derived {num_shards} shards for size {size_in_gb}GB")

            model_with_head = self.model_unwrapped
            outputs = model_with_head.model(**batch, use_cache=False)
            hidden_states = outputs.last_hidden_state

            kwargs_to_shard = dict(
                hidden_states=hidden_states,
                shift_labels=shift_labels,
            )
            kwargs_to_pass = dict(model_with_head=model_with_head, vocab_size=self.model_unwrapped.config.vocab_size)
            grad_requiring_tensor_key = "hidden_states"
            compute_params = [model_with_head.lm_head.weight]
            seqlen = shift_labels.shape[1]

            # since -100s shift_labels are ignored we have to perform a weighted average on each
            # loss slice as each slice may contribute a different number of non- -100 labels
            def fused_logits_loss_fn(
                model_with_head=None, hidden_states=None, labels=None, shift_labels=None, vocab_size=0
            ):
                logits = model_with_head.lm_head(hidden_states)
                if all((shift_labels == -100).squeeze()):
                    # fake loss calculation, since CE will return nan, but grads will be set
                    # a normal loss_fn upcasts logits to float so match it
                    loss_sum = (logits.sum() * 0.0).float()
                else:
                    good_items = sum((shift_labels != -100).squeeze())
                    loss = model_with_head.loss_function(
                        logits=logits, labels=labels, vocab_size=vocab_size, shift_labels=shift_labels
                    )
                    loss_sum = loss * good_items
                return loss_sum

            total_loss_sum = sequence_tiled_compute(
                fused_logits_loss_fn,
                seqlen,
                num_shards,
                kwargs_to_shard,
                kwargs_to_pass,
                grad_requiring_tensor_key,
                compute_params,
                output_unshard_dimension=0,  # loss is a scalar
                output_reduction="sum",
            )
            total_good_items = sum((shift_labels != -100).squeeze())
            loss = total_loss_sum / total_good_items

        # differentiable weighted per-shard-loss aggregation across ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=self.sp_group)
        good_tokens = sum((shift_labels != -100).view(-1))
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=self.sp_group)
        total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(self.sp_world_size))
        total_good_tokens = sum(good_tokens_per_rank)
        loss = total_loss / total_good_tokens

        return loss
