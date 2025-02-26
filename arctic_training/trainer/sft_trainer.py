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

from typing import Dict
from typing import Union

import torch

from arctic_training.checkpoint.ds_engine import DSCheckpointEngine
from arctic_training.checkpoint.hf_engine import HFCheckpointEngine
from arctic_training.data.sft_factory import SFTDataFactory
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.model.liger_factory import LigerModelFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.scheduler.hf_factory import HFSchedulerFactory
from arctic_training.tokenizer.hf_factory import HFTokenizerFactory
from arctic_training.trainer.trainer import Trainer
from arctic_training.debug import print_rank0, print_rank, exit

def to_device(batch: Dict, device: str) -> Dict:
    output = {}
    for k, v in batch.items():
        output[k] = v.to(device)
    return output


class SFTTrainer(Trainer):
    name = "sft"
    data_factory: SFTDataFactory
    model_factory: Union[HFModelFactory, LigerModelFactory]
    checkpoint_engine: Union[DSCheckpointEngine, HFCheckpointEngine]
    optimizer_factory: Union[FusedAdamOptimizerFactory]
    scheduler_factory: Union[HFSchedulerFactory]
    tokenizer_factory: Union[HFTokenizerFactory]

    def loss(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)

        if self.config.sequence_parallel_size == 1:
            outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
        else:

            # gather DL batches into super-batches
            # XXX: at the moment assuming DL gives us a nice max_length chunks that are already padded
            # for the general case may need to massage the concatenated DL samples and remove padding and then repad at the end.
            from deepspeed.utils import groups
            import torch
            import deepspeed.comm as dist
            sp_group = groups._get_sequence_parallel_group()
            sp_world_size = groups._get_sequence_parallel_world_size()
            sp_rank = groups._get_sequence_parallel_rank()

            from collections import defaultdict
            micro_batches = defaultdict(dict)
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
                print_rank0(f"before gather: {k}: {batch[k].shape=}")
                print_rank0(f"before gather: {k}: {batch[k]=}")
                with torch.no_grad():
                    tensor_list = [torch.zeros_like(batch[k]) for _ in range(self.config.sequence_parallel_size)]
                    dist.all_gather(tensor_list, batch[k], group=sp_group)
                    # gathering on the data dimension
                    # will be concatenating and later splitting again for the more general case
                    # batch[k] = torch.cat(tensor_list, dim=1)
                    for rank, tensor in enumerate(tensor_list):
                        micro_batches[rank][k] = tensor
                print_rank0(f"after gather: {k}: {batch[k].shape=}")
                print_rank0(f"after gather: {k}: {batch[k]=}")

            loss_aggregate = 0
            # we need to chunk twice - each time on SP size level
            # - the first time is because we artifically made the seqlen SP-times longer
            # - the second time is because of the Ulysses algorithm

            self.model.set_gradient_accumulation_boundary(False)


            for sub_step_id in range(self.config.sequence_parallel_size):
                batch = micro_batches[sub_step_id]

                print_rank0(batch)

                # XXX: probably need to do padding so that all sequence chunks are the same?!
                import math
                print_rank0(f"{len(batch['input_ids'][0])=}")
                seq_length = self.config.data.max_length

                chunk_len = math.ceil(seq_length / sp_world_size)
                print_rank0(f"{seq_length=}")
                print_rank0(f"{chunk_len=}")

                for k in batch.keys():
                    # we are not chunking labels!
                    if k in ["input_ids", "position_ids"]:
                        batch[k] = batch[k][:, chunk_len*sp_rank:chunk_len*(sp_rank+1)].to(self.device)
                    else:
                        batch[k] = batch[k].to(self.device)

                    print_rank0(f"after sp: {k}: {batch[k].shape=}")
                    print_rank0(f"after sp: {k}: {batch[k]=}")
                #outputs = self.model(**batch, use_cache=False)
                #loss = outputs.loss

                # XXX: this would be the same not just for SFT so probably should abstract it away
                from deepspeed.utils import groups
                import torch.distributed as dist
                import torch

                # because we have to gather logits from all sp ranks we have to do the loss function ourselves
                # therefore remove labels to avoid an attempt to calculate loss by transformers
                labels = batch.pop("labels")
                outputs = self.model(**batch, use_cache=False)

                logits = outputs.logits
                print_rank(f"{torch.norm(logits)=}")
                print_rank(f"{logits.shape=}")
                #print_rank(f"{logits.dtype=}")
                print_rank(f"{labels.shape=}")

                # XXX: stick into the trainer object
                sp_group = groups._get_sequence_parallel_group()
                sp_world_size = groups._get_sequence_parallel_world_size()
                # we need the differentiable all_gather, which is the functional version of it
                import torch.distributed.nn.functional
                tensor_list = torch.distributed.nn.functional.all_gather(logits, sp_group)
                # concatenate on the seqlen dimension
                logits = torch.cat(tensor_list, dim=1)
                print_rank(f"after cat: {logits.shape=}")

                loss = self.model_unwrapped.loss_function(logits=logits, labels=labels, vocab_size=self.model_unwrapped.config.vocab_size) #/ sp_world_size
                print_rank0(f"intermediary {loss.item()*sp_world_size=}")



                #loss = self.loss(batch)
                loss_aggregate += loss.item()*sp_world_size

                print_rank(f"{self.train_batch_idx}-{sub_step_id}: {loss.requires_grad=}")
                print_rank(f"{self.train_batch_idx}-{sub_step_id}: {loss=}")

                avg_loss = self.model.backward(loss)
                print_rank0(f"zero loss: {avg_loss}")

                # from deepspeed.utils import safe_get_full_grad
                # print_rank(f"{torch.norm(safe_get_full_grad(self.model.module.lm_head.weight))=}")
                # print_rank(f"{torch.norm(safe_get_full_grad(self.model.module.model.layers[0].self_attn.q_proj.weight))=}")

                # #w = self.model.module.model.layers[0].self_attn.q_proj.weight
                # w = self.model.module.lm_head.weight
                from deepspeed.utils import safe_get_full_grad
                print_rank0(f"{torch.norm(safe_get_full_grad(self.model.module.lm_head.weight))=}")
                print_rank0(f"{torch.norm(safe_get_full_grad(self.model.module.model.layers[0].self_attn.q_proj.weight))=}")

            self.model.set_gradient_accumulation_boundary(True)


        return loss

    # # possible future version of integrated loss - would need to move the rest of the code into all_gather_logits_and_do_loss
    # def loss(self, batch) -> torch.Tensor:
    #     batch = to_device(batch, self.device)

    #     if self.config.sequence_parallel_size != 1:
    #         # because we have to gather logits from all sp ranks we have to do the loss function ourselves
    #         # therefore remove labels to avoid an attempt to calculate loss by transformers and store those for later use
    #         labels = batch.pop("labels")

    #     outputs = self.model(**batch, use_cache=False)

    #     if self.config.sequence_parallel_size == 1:
    #         loss = outputs.loss
    #     else:
    #         loss = all_gather_logits_and_do_loss(outputs.logits, labels)
    #     return loss
