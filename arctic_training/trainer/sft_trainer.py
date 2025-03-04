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
from arctic_training.debug import print_rank0, print_rank, exit, see_memory_usage

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

        import torch
        if self.config.sequence_parallel_size == 1:
            # XXX: weird
            batch["labels"] = batch["labels"].type(torch.LongTensor)
            outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss
        else:

            # ensure shapes are correct
            if not (batch["input_ids"].shape == batch["position_ids"].shape == batch["labels"].shape):
                raise ValueError(f'Borked batch {batch["input_ids"].shape=} != {batch["position_ids"].shape=} != {batch["labels"].shape=}) in DataLoader->iter->next, cannot continue with Sequence parallelism')

            # gather DL batches into super-batches
            # Important: DL doesn't always yield max_length batches. Different ranks may have different seqlen and each could be <= max_length (but always divisible by 256)
            from deepspeed.utils import groups
            import torch
            # XXX: should it be torch.dist? when to use ds dist?
            import deepspeed.comm as dist
            sp_group = groups._get_sequence_parallel_group()
            sp_world_size = groups._get_sequence_parallel_world_size()
            sp_rank = groups._get_sequence_parallel_rank()

            from collections import defaultdict
            micro_batches = defaultdict(dict)
            # Efficient gathering of batch inputs across ranks:
            # The problem is that our DL doesn't guarantee the same seqlen on all ranks and may give, 3x 1024 and 1x 768 on 4 gpus for max_length 1024. so 3 options we have to be able to gather batches are:
            # 1. use all_gather_object - which allows different shapes - but potentially introducing an undesired overhead - 2x pickle calls
            # 2. use all_gather and change DL pad to make sure that all ranks always get the same input shape - this creates its own overhead since if we say have ranks with seqlen 512, 768, 1024, 1024 - now we will need to process 4x 1024 seqlens
            # 3. use all_gather and post gathering truncate tensors to their intended length - another overhead of allocating and truncating tensors
            # using approach (1) for now but might want to benchmark later the other 2 approaches

            see_memory_usage("before gathering", force=True)

            # XXX: if using all_gather_object we can gather the whole batch at once and not per-key! so can drop the loop for that approach
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
                print_rank(f"before gather: {k}: {batch[k].shape=}", skip=False)
                #print_rank0(f"before gather: {k}: {batch[k]=}")
                with torch.no_grad():
                    # tensor_list = [torch.zeros_like(batch[k]) for _ in range(sp_world_size)]
                    # dist.all_gather(tensor_list, batch[k], group=sp_group)
                    tensor_list = [None for _ in range(sp_world_size)]
                    torch.distributed.all_gather_object(tensor_list, batch[k])
                    # gathering on the data dimension
                    # will be concatenating and later splitting again for the more general case
                    # batch[k] = torch.cat(tensor_list, dim=1)
                    for rank, tensor in enumerate(tensor_list):
                        micro_batches[rank][k] = tensor
                        print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k].shape=}", skip=False)
                        #print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k].device=}", skip=False)
                        print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k]=}", skip=False)
            #exit()
            loss_aggregate = 0
            # we need to chunk twice - each time on SP size level
            # - the first time is because we artifically made the seqlen SP-times longer
            # - the second time is because of the Ulysses algorithm

            see_memory_usage("after gathering", force=True)
            self.model.set_gradient_accumulation_boundary(False)

            for sub_step_id in range(sp_world_size):
                batch = micro_batches[sub_step_id]

                see_memory_usage(f"{sub_step_id=} start", force=True)
                #print_rank0(batch)

                # XXX: probably need to do padding so that all sequence chunks are the same?!
                import math
                print_rank0(f"{sub_step_id}: {len(batch['input_ids'][0])=}")
                seq_length = len(batch['input_ids'][0])
                #seq_length = self.config.data.max_length

                if seq_length % sp_world_size != 0:
                    raise ValueError(f"{sub_step_id=}: batch's seqlen={seq_length} isn't divisible by sp-size={sp_world_size}")
                ##chunk_len = math.ceil(seq_length / sp_world_size)
                chunk_len = int(seq_length / sp_world_size)
                print_rank0(f"{sub_step_id=}: {seq_length=}")
                print_rank0(f"{sub_step_id=}: {chunk_len=}")

                for k in batch.keys():
                    # we are not chunking labels!
                    if k in ["input_ids", "position_ids"]:
                        print_rank(f"SLICING {k} {chunk_len=}: {sp_rank=}", skip=False)
                        batch[k] = batch[k][:, chunk_len*sp_rank:chunk_len*(sp_rank+1)].to(self.device)
                    else:
                        print_rank(f"KEEPING {k} {batch[k].shape=}", skip=False)
                        batch[k] = batch[k].to(self.device)

                    #print_rank0(f"after sp: {k}: {batch[k].shape=}")
                    #print_rank0(f"after sp: {k}: {batch[k]=}")
                #outputs = self.model(**batch, use_cache=False)
                #loss = outputs.loss
                see_memory_usage(f"{sub_step_id=} after chunking", force=True)

                # XXX: this would be the same not just for SFT so probably should abstract it away
                from deepspeed.utils import groups
                import torch.distributed as dist
                import torch

                # because we have to gather logits from all sp ranks we have to do the loss function ourselves
                # therefore remove labels to avoid an attempt to calculate loss by transformers
                labels = batch.pop("labels")
                #labels = labels.type(torch.LongTensor)

                see_memory_usage(f"{sub_step_id=} before forward", force=True)

                outputs = self.model(**batch, use_cache=False)
                see_memory_usage(f"{sub_step_id=} after forward", force=True)

                logits = outputs.logits
                #print_rank(f"{sub_step_id=}: {torch.norm(logits)=}")
                #print_rank(f"{sub_step_id=}: {logits.shape=}")
                #print_rank(f"{logits.dtype=}")
                #print_rank(f"{sub_step_id=}: {labels.shape=}")

                # XXX: stick into the trainer object
                #sp_group = groups._get_sequence_parallel_group()
                #sp_world_size = groups._get_sequence_parallel_world_size()
                # we need the differentiable all_gather, which is the functional version of it
                import torch.distributed.nn.functional
                tensor_list = torch.distributed.nn.functional.all_gather(logits, sp_group)
                # concatenate on the seqlen dimension
                logits = torch.cat(tensor_list, dim=1)
                del tensor_list
                print_rank(f"after cat: {logits.shape=}")
                see_memory_usage(f"{sub_step_id=} after cat", force=True)

                #print_rank(f"LOSS {logits.shape=}: {labels.shape=}", skip=False)

                loss = self.model_unwrapped.loss_function(logits=logits, labels=labels, vocab_size=self.model_unwrapped.config.vocab_size)
                #print_rank0(f"intermediary {loss.item()*sp_world_size=}")
                see_memory_usage(f"{sub_step_id=} after loss", force=True)

                # optimize memory
                del logits
                del labels

                #loss = self.loss(batch)
                loss_aggregate += loss.item()*sp_world_size

                print_rank(f"{self.train_batch_idx}-{sub_step_id}: {loss.requires_grad=}")
                print_rank(f"{self.train_batch_idx}-{sub_step_id}: {loss=}")

                avg_loss = self.model.backward(loss)
                print_rank0(f"zero loss: {avg_loss}")
                see_memory_usage(f"{sub_step_id=} after backward", force=True)


                # from deepspeed.utils import safe_get_full_grad
                # print_rank(f"{torch.norm(safe_get_full_grad(self.model.module.lm_head.weight))=}")
                # print_rank(f"{torch.norm(safe_get_full_grad(self.model.module.model.layers[0].self_attn.q_proj.weight))=}")

                # #w = self.model.module.model.layers[0].self_attn.q_proj.weight
                # w = self.model.module.lm_head.weight
                #from deepspeed.utils import safe_get_full_grad
                #print_rank0(f"{torch.norm(safe_get_full_grad(self.model.module.lm_head.weight))=}")
                #print_rank0(f"{torch.norm(safe_get_full_grad(self.model.module.model.layers[0].self_attn.q_proj.weight))=}")

            self.model.set_gradient_accumulation_boundary(True)

        # XXX: temp to measure the real memory usage
        # gc_empty_cuda_cache()

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
