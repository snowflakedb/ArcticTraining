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
from arctic_training.optimizer.adam_factory import CPUAdamOptimizerFactory
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


# start block to remove on enabling of API_change_36607
import torch.nn as nn
def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

def ForCausalLMLossSharded(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, shift_labels_pad_index: int = None, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    if shift_labels_pad_index is None:
        shift_labels_pad_index = ignore_index
    labels = nn.functional.pad(labels, (0, 1), value=shift_labels_pad_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss
# end block to remove on enabling of API_change_36607

# class FakeLoss(torch.autograd.Function):
#     """

#     """

#     @staticmethod
#     def forward(ctx, logits):
#         loss = logits.sum() * 0.0
#         ctx.logits_shape = logits.shape
#         ctx.device = logits.device
#         ctx.dtype = logits.dtype
#         return loss

#     @staticmethod
#     def backward(ctx, output_gracds):
#         logits_grads = torch.zeros(ctx.logits_shape, device=ctx.device, dtype=ctx.dtype)
#         return logits_grads, None, None


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

        import torch
        if self.config.sequence_parallel_size == 1:
            # XXX: weird
            batch["labels"] = batch["labels"].type(torch.LongTensor)
            outputs = self.model(**batch, use_cache=False)

            logits = outputs.logits
            # print_rank(f"{logits=}", skip=False)
            # print_rank(f"logit nans: {torch.isnan(logits).sum()}", skip=False)
            # print_rank(f"logit infs: {torch.isinf(logits).sum()}", skip=False)

            loss = outputs.loss
            self.model.backward(loss)

            import torch.distributed.nn.functional
            with torch.no_grad():
                # average losses
                losses_per_rank = torch.distributed.nn.functional.all_gather(loss)
                print(f"LOSS {losses_per_rank=}")
                average_loss = torch.cat([l.unsqueeze(0) for l in losses_per_rank], dim=0).mean()
                print(f"LOSS {average_loss=}")

            return average_loss

        else:

            #print_rank(f"YYYY {batch['position_ids']=}", skip=False)
            #exit()
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
            #print_rank(f"{self.tokenizer.decode(batch['input_ids'][0])=}", skip=False)
            #exit()

            # XXX: if using all_gather_object we can gather the whole batch at once and not per-key! so can drop the loop for that approach
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
                print_rank(f"before gather: {k}: {batch[k].shape=}", skip=False)
                #print_rank0(f"before gather: {k}: {batch[k]=}")
                with torch.no_grad():
                    # tensor_list = [torch.zeros_like(batch[k]) for _ in range(sp_world_size)]
                    # dist.all_gather(tensor_list, batch[k], group=sp_group)
                    tensor_list = [None for _ in range(sp_world_size)]
                    torch.distributed.all_gather_object(tensor_list, batch[k], group=sp_group)
                    # gathering on the data dimension
                    # will be concatenating and later splitting again for the more general case
                    # batch[k] = torch.cat(tensor_list, dim=1)
                    for rank, tensor in enumerate(tensor_list):
                        micro_batches[rank][k] = tensor
                        print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k].shape=}", skip=False)
                        #print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k].device=}", skip=False)
                        #print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k]=}", skip=False)
                        # if k == "input_ids":
                        #     print_rank0(f"{self.tokenizer.decode(micro_batches[rank][k][0])=}", skip=False)

            #exit()
            loss_aggregate = 0
            # we need to chunk twice - each time on SP size level
            # - the first time is because we artifically made the seqlen SP-times longer
            # - the second time is because of the Ulysses algorithm

            see_memory_usage("after gathering", force=True)
            self.model.set_gradient_accumulation_boundary(False)

            losses = []
            for sub_step_id in range(sp_world_size):

                # if sub_step_id == 1:
                #     continue
                # if sub_step_id == 3:
                #     break


                batch = micro_batches[sub_step_id]

                see_memory_usage(f"{sub_step_id=} start", force=True)
                #print_rank0(batch)

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

                # XXX: API_change_36607: this API will work once https://github.com/huggingface/transformers/pull/36607 is merged and a new transformers is released and we require that version or higher - then can remove all the `non API_change_36607` code branches
                API_change_36607 = True
                if not API_change_36607:
                    # get the first label elem of each shard to be later used in the loss function
                    # the last rank will have -100 instead
                    shift_labels_pad_index = {}
                    for rank in range(sp_world_size):
                        # XXX: careful - assuming bs=1 - need to generalize this for bs>1
                        if rank+1 == sp_world_size:
                            index = -100
                        else:
                            index = batch["labels"][0][chunk_len*(rank+1)].item()
                        shift_labels_pad_index[rank] = index
                    print_rank(f"{shift_labels_pad_index=}", skip=False)
                    #exit()

                # to enable the correct mean calculation across shards before sharding the micro batch:
                # 1. count the number of non- `-100`` elements per shard
                # 2. and subtract one more element because of label shifting
                non_skipped_items = {}
                for rank in range(sp_world_size):
                    non_skipped = (batch["labels"][:, chunk_len*rank:chunk_len*(rank+1)] != -100).sum().item()
                    if non_skipped > 1:
                        non_skipped -= 1
                    non_skipped_items[rank] = non_skipped
                print_rank(f"{non_skipped_items=}", skip=False)

                if API_change_36607:
                    # because we have to gather logits from all sp ranks we have to do the loss function ourselves
                    # therefore remove labels to avoid an attempt to calculate loss by transformers
                    labels = batch.pop("labels")
                    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
                    batch["shift_labels"] = labels[..., 1:].contiguous()
                    # free up temp memory
                    del labels

                # batch sharding
                for k in batch.keys():
                    print_rank(f"SLICING {k} {chunk_len=}: {sp_rank=}", skip=False)
                    batch[k] = batch[k][:, chunk_len*sp_rank:chunk_len*(sp_rank+1)].to(self.device)
                    # else:
                    #     print_rank(f"KEEPING {k} {batch[k].shape=}", skip=False)
                    #     batch[k] = batch[k].to(self.device)

                    print_rank0(f"after sp: {k}: {batch[k].shape=}")
                    #print_rank0(f"after sp: {k}: {batch[k]=}")
                #outputs = self.model(**batch, use_cache=False)
                #loss = outputs.loss
                see_memory_usage(f"{sub_step_id=} after chunking", force=True)

                # XXX: this would be the same not just for SFT so probably should abstract it away
                from deepspeed.utils import groups
                import torch.distributed as dist
                import torch

                if not API_change_36607:
                    # because we have to gather logits from all sp ranks we have to do the loss function ourselves
                    # therefore remove labels to avoid an attempt to calculate loss by transformers
                    labels = batch.pop("labels")
                    labels = labels.type(torch.LongTensor)

                    #print_rank0(f"after sp: {k}: {batch[k].shape=}")
                    #print_rank0(f"after sp: {k}: {batch[k]=}")
                see_memory_usage(f"{sub_step_id=} before forward", force=True)

                print_rank(f"SLICE DECODE: {sub_step_id=} {self.tokenizer.decode(batch['input_ids'][0])=}", skip=False)
                print_rank(f"SLICE DECODE: {sub_step_id=} {batch['position_ids'][0]=}", skip=False)
                # if API_change_36607:
                #     print_rank(f"SLICE DECODE: {sub_step_id=} {batch['shift_labels'][0]=}", skip=False)
                # else:
                #     print_rank(f"SLICE DECODE: {sub_step_id=} {batch['labels'][0]=}", skip=False)

                if API_change_36607:
                    shift_labels = batch.pop("shift_labels")
                    #print_rank(f"{shift_labels=}", skip=False)
                    see_memory_usage(f"{sub_step_id=} after shift labels", force=True)

                outputs = self.model(**batch, use_cache=False)
                logits = outputs.logits

                see_memory_usage(f"{sub_step_id=} after forward", force=True)

                #print_rank(f"{labels=}", skip=False)
                #print_rank(f"{logits=}", skip=False)
                # print_rank(f"logit nans: {torch.isnan(logits).sum()}", skip=False)
                # print_rank(f"logit infs: {torch.isinf(logits).sum()}", skip=False)


                if (not API_change_36607 and all((labels == -100).squeeze())) or (API_change_36607 and all((shift_labels == -100).squeeze())):
                    # this is the case where all labels in the micro-batch are -100 (very common for SFT) - CE returns `nan` in this case, so we don't want to call loss and instead create a differentiable loss `0` which will also set all the grads to `0` in `backward` - the effect of this is akin to a perfect score where the model needs no adjustment since grads will be all zeros.
                    loss = (logits.sum() * 0.0).float()
                    #loss = FakeLoss.apply(logits)
                else:

                    # XXX: API-change-36607: this API will work once https://github.com/huggingface/transformers/pull/36607 is merged and a new transformers is released and we require that version or higher
                    if API_change_36607:
                        loss = self.model_unwrapped.loss_function(logits=logits, labels=None, vocab_size=self.model_unwrapped.config.vocab_size, shift_labels=shift_labels)
                    else:
                        # deal with the boundary issue which would lose one label element per rank
                        #
                        # In unsharded string we end up with (shift left):
                        #
                        # input_ids: [1 2 3 4 5 6 7    8   ]
                        # labels   : [1 2 3 4 5 6 7    8   ]
                        # shiftedl : [2 3 4 5 6 7 8 -100]
                        #
                        # when sharded we lose label 5 once shifted:
                        #
                        # input_ids: [1 2 3    4] [5 6 7    8]
                        # labels   : [1 2 3    4] [5 6 7    8]
                        # shiftedl : [2 3 4 -100] [6 7 8 -100]
                        #
                        # since HF doesn't let us pre-shift labels - we use this workaround:
                        # - for all but the last shards: add another column of elements so that logits[-1] is random and labels[-1] is the first labels[0] of the next shard
                        # - for the last shard keep things the same
                        # so that it looks like:
                        #
                        # input_ids: [1 2 3 4    r] [5 6 7    8]
                        # labels   : [1 2 3 4    5] [5 6 7    8]
                        # shiftedl : [2 3 4 5 -100] [6 7 8 -100]
                        #
                        # where r = a random [bs, 1, vocab-size] element
                        #
                        # which is the same as our original unsharded sequence
                        #
                        # input_ids: [1 2 3 4 5 6 7    8]
                        # labels:    [1 2 3 4 5 6 7    8]
                        # shiftedl : [2 3 4 5 6 7 8 -100]
                        #
                        # the other solution is to replace -100 padding with the first label of the next shard, so then we get:
                        #
                        # input_ids: [1 2 3 4]  [5 6 7 8]
                        # labels   : [1 2 3 4]  [5 6 7 8]
                        # shiftedl : [2 3 4 5]  [6 7 8 -100]

                        # print_rank(f"b {labels.shape=}", skip=False)
                        # print_rank(f"b {logits.shape=}", skip=False)
                        # if sp_rank+1 != sp_world_size:
                        #     # XXX: need to make shift_labels_pad_index work for bs>1 as it can be a different value for each bs entry
                        #     #labels = torch.nn.functional.pad(labels, (0, 1), value=shift_labels_pad_index[sp_rank])
                        #     labels = torch.cat((labels, torch.tensor(shift_labels_pad_index[sp_rank])[..., None, None]), dim=1)
                        #     logits = torch.cat((logits, torch.zeros_like(logits[:,0,:])[:, None]), dim=1)

                        # print_rank(f"a {labels.shape=}", skip=False)
                        # print_rank(f"a {logits.shape=}", skip=False)

                        loss = ForCausalLMLossSharded(logits=logits,
                                                      labels=labels,
                                                      shift_labels_pad_index=shift_labels_pad_index[sp_rank],
                                                      vocab_size=self.model_unwrapped.config.vocab_size)


                    # loss = self.model_unwrapped.loss_function(logits=logits, labels=labels, vocab_size=self.model_unwrapped.config.vocab_size)

                #loss = outputs.loss
                print_rank(f"LOSS local {loss=}", skip=False)

                # free up temp mem (e.g. outputs.logits are huge)
                del outputs

                see_memory_usage(f"{sub_step_id=} before gathered loss", force=True)
                #exit()

                # if torch.isnan(loss):
                #     break
                #     #continue
                #     #loss = torch.tensor(0.0).to(self.device).requires_grad_() + 0.0
                # differentiable loss aggregation across ranks
                import torch.distributed.nn.functional
                #loss = torch.distributed.nn.functional.all_reduce(loss, op=torch.distributed.ReduceOp.SUM, group=sp_group)
                losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
                #print(f"LOSS {losses_per_rank=}")
                print_rank(f"LOSS {losses_per_rank=}", skip=False)

                # # filter out nans
                # losses_per_rank = [l for l in losses_per_rank if not torch.isnan(l)]
                # if len(losses_per_rank) == 0 :
                #     # XXX: weird that all ranks get it sometimes even though labels are fine - bad data? skipping as a bad batch for now
                #     continue


                # since each shard may have a variable number of skipped elemented - need to calculate a weighted mean depending on each rank's contribution - this will also take care of loss=0 when all elements are -100 in a shard
                # XXX: not expecting a total of 0-non-skipped items for div
                loss = sum(losses_per_rank[rank] * non_skipped_items[rank] for rank in range(sp_world_size)) / sum(non_skipped_items.values())
                # this is a much simpler version w/o weighting
                # skip 0.0 entries when calculating total loss per batch
                # loss = torch.stack(list(l for l in losses_per_rank if l != 0)).mean()

                #loss = torch.cat([l.unsqueeze() for l in losses_per_rank], dim=0).mean()
                #loss = sum(loss_per_rank) # / sp_world_size
                #loss = sum(tensor_list)
                print_rank(f"LOSS averaged {loss=}", skip=False)
                print("LOSS", loss)
                see_memory_usage(f"{sub_step_id=} after gathered loss", force=True)

                #exit()

                #logits = outputs.logits
                #print_rank(f"{sub_step_id=}: {torch.norm(logits)=}", skip=False)
                #print_rank(f"{sub_step_id=}: {logits.shape=}")
                #print_rank(f"{logits.dtype=}")
                #print_rank(f"{sub_step_id=}: {labels.shape=}")

                # # XXX: stick into the trainer object
                # #sp_group = groups._get_sequence_parallel_group()
                # #sp_world_size = groups._get_sequence_parallel_world_size()
                # # we need the differentiable all_gather, which is the functional version of it
                # import torch.distributed.nn.functional
                # tensor_list = torch.distributed.nn.functional.all_gather(logits, sp_group)
                # # concatenate on the seqlen dimension
                # logits = torch.cat(tensor_list, dim=1)
                # del tensor_list
                # print_rank(f"after cat: {logits.shape=}")
                # see_memory_usage(f"{sub_step_id=} after cat", force=True)

                #print_rank(f"LOSS {logits.shape=}: {labels.shape=}", skip=False)

                # loss = self.model_unwrapped.loss_function(logits=logits, labels=labels, vocab_size=self.model_unwrapped.config.vocab_size)
                # #print_rank0(f"intermediary {loss.item()*sp_world_size=}")

                # # optimize memory
                # del logits
                # del labels

                # #loss = self.loss(batch)
                # loss_aggregate += loss.item()*sp_world_size

                #print_rank(f"{self.train_batch_idx}-{sub_step_id}: {loss.requires_grad=}")
                print_rank(f"{self.train_batch_idx}-{sub_step_id}: {loss=}")

                see_memory_usage(f"{sub_step_id=} before backward", force=True)
                self.model.backward(loss)
                # print_rank(f"{labels[0][70:80]=}", skip=False)
                # print_rank(f"{logits[0][70:80]=}", skip=False)
                # print_rank(f'{batch["input_ids"][0][70:80]=}', skip=False)
                # print_rank(f'{batch["input_ids"].grad[0][70:80]=}', skip=False)
                # print_rank(f"{logits.grad[0][70:80]=}", skip=False)
                # exit()

                print_rank0(f"zero loss: {loss}", skip=False)
                # print_rank0(f"zero loss: {avg_loss}", skip=False)
                see_memory_usage(f"{sub_step_id=} after backward", force=True)

                losses.append(loss.detach().item())


                # from deepspeed.utils import safe_get_full_grad
                # print_rank(f"{torch.norm(safe_get_full_grad(self.model.module.lm_head.weight))=}")
                # print_rank(f"{torch.norm(safe_get_full_grad(self.model.module.model.layers[0].self_attn.q_proj.weight))=}")

                # #w = self.model.module.model.layers[0].self_attn.q_proj.weight
                # w = self.model.module.lm_head.weight
                #from deepspeed.utils import safe_get_full_grad
                #print_rank0(f"{torch.norm(safe_get_full_grad(self.model.module.lm_head.weight))=}")
                #print_rank0(f"{torch.norm(safe_get_full_grad(self.model.module.model.layers[0].self_attn.q_proj.weight))=}")

            self.model.set_gradient_accumulation_boundary(True)

            # for per-iteration reporting
            if len(losses) == 0:
                loss = float('nan')
            else:
                loss = sum(losses) / len(losses)

        #exit()
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
