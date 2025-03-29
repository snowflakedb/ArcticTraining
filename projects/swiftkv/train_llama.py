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

from typing import Any
from typing import Union

import llama_swiftkv
import torch
import torch.nn.functional as F
from deepspeed.runtime.zero import GatheredParameters

from arctic_training import HFCheckpointEngine
from arctic_training import HFModelFactory
from arctic_training import ModelConfig
from arctic_training import SFTTrainer
from arctic_training import TrainerConfig
from arctic_training import logger
from arctic_training.trainer.sft_trainer import to_device


llama_swiftkv.register_auto()


class SwiftKVModelConfig(ModelConfig):
    num_key_value_layers: int
    key_value_group_size: int = 1


class SwiftKVModelFactory(HFModelFactory):
    name = "swiftkv"
    config: SwiftKVModelConfig

    def post_create_config_callback(self, hf_config):
        llama_swiftkv.register_auto()

        config_dict = hf_config.to_dict()
        hf_config = llama_swiftkv.LlamaSwiftKVConfig.from_dict(config_dict)

        hf_config.num_key_value_layers = self.config.num_key_value_layers
        hf_config.key_value_group_size = self.config.key_value_group_size

        return hf_config

    def post_create_model_callback(self, model):
        # Freeze all teacher parameters
        for param in model.parameters():
            param.requires_grad = False

        # Initialize student layers
        model.model.norm_swiftkv.weight.requires_grad = True
        for layer_idx in range(model.config.num_key_value_layers,
                               model.config.num_hidden_layers):
            layer = model.model.layers[layer_idx]
            if not model.config.swiftkv:
                # Initialize q_proj_swiftkv
                logger.info(
                    f"Initializing q_proj_swiftkv for layer {layer_idx}"
                )
                with GatheredParameters(layer.parameters(), modifier_rank=0):
                    layer.self_attn.q_proj_swiftkv.weight.data.copy_(
                        layer.self_attn.q_proj.weight.data
                    )
            layer.self_attn.q_proj_swiftkv.weight.requires_grad = True
        for layer_idx in range(
            model.config.num_key_value_layers,
            model.config.num_hidden_layers,
            model.config.key_value_group_size,
        ):
            this_attn = model.model.layers[layer_idx].self_attn
            next_attn = [
                model.model.layers[layer_idx + i].self_attn
                for i in range(model.config.key_value_group_size)
            ]
            for param in ("k_proj", "v_proj"):
                weights = [getattr(this_attn, f"{param}_swiftkv").weight] + [
                    getattr(attn, f"{param}").weight for attn in next_attn
                ]
                if not model.config.swiftkv:
                    # Initialize k_proj_swiftkv and v_proj_swiftkv
                    logger.info(
                        f"Initializing {param}_swiftkv for layer {layer_idx}"
                    )
                    with GatheredParameters(weights, modifier_rank=0):
                        weights[0].data.copy_(
                            sum(weights[1:]) / model.config.key_value_group_size
                        )
                getattr(this_attn, f"{param}_swiftkv").weight.requires_grad = True
        model.gradient_checkpointing_enable()
        return model


class SwiftKVTrainerConfig(TrainerConfig):
    decoder_loss_mult: float = 0.0
    temperature: float = 1.0


class SwiftKVTrainer(SFTTrainer):
    name = "swiftkv"
    config: SwiftKVTrainerConfig
    model_factory: SwiftKVModelFactory
    checkpoint_engine: Union[HFCheckpointEngine]

    def loss(self, batch: Any) -> torch.Tensor:
        batch = to_device(batch, self.device)

        with torch.no_grad():
            self.model.swiftkv(False)
            self.model.eval()
            teacher_outputs = self.model(
                **batch,
                output_hidden_states=(self.config.decoder_loss_mult > 0),
            )

        self.model.swiftkv(True)
        self.model.train()
        student_outputs = self.model(
            **batch,
            output_hidden_states=(self.config.decoder_loss_mult > 0),
        )

        distill_loss = self.distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            temperature=self.config.temperature,
        )

        decoder_loss = torch.zeros_like(distill_loss)
        if self.config.decoder_loss_mult > 0:
            decoder_loss_count = 0
            for layer_idx in [15, 23]:
                student_hidden = student_outputs.hidden_states[layer_idx]
                teacher_hidden = teacher_outputs.hidden_states[layer_idx]
                decoder_loss += torch.linalg.norm(
                    student_hidden - teacher_hidden,
                    dim=-1,
                ).mean()
                decoder_loss_count += 1
            decoder_loss *= self.config.decoder_loss_mult / decoder_loss_count

        logger.info(
            f"student loss: {student_outputs.loss.item()}, teacher loss:"
            f" {teacher_outputs.loss.item()}, distill loss: {distill_loss.item()}"
        )

        loss = distill_loss + decoder_loss

        torch.cuda.synchronize()

        return loss

    def sp_fwd_bwd_loss(self, batch) -> torch.Tensor:
        from arctic_training.debug import print_rank0, print_rank, exit, see_memory_usage

        batch = to_device(batch, self.device)

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

        see_memory_usage("before gathering", force=False)
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

        see_memory_usage("after gathering", force=False)
        self.model.set_gradient_accumulation_boundary(False)

        losses = []
        for sub_step_id in range(sp_world_size):
            #print(f"{sub_step_id=}")
            # if sub_step_id == 1:
            #     continue
            # if sub_step_id == 3:
            #     break


            batch = micro_batches[sub_step_id]

            see_memory_usage(f"{sub_step_id=} start", force=False)
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
            see_memory_usage(f"{sub_step_id=} after chunking", force=False)

            # XXX: this would be the same not just for SFT so probably should abstract it away
            from deepspeed.utils import groups
            import torch.distributed as dist
            import torch

            see_memory_usage(f"{sub_step_id=} before forward", force=False)

            print_rank(f"SLICE DECODE: {sub_step_id=} {self.tokenizer.decode(batch['input_ids'][0])=}", skip=False)
            print_rank(f"SLICE DECODE: {sub_step_id=} {batch['position_ids'][0]=}", skip=False)

            shift_labels = batch.pop("shift_labels")
            #print_rank(f"{shift_labels=}", skip=False)
            see_memory_usage(f"{sub_step_id=} after shift labels", force=False)

            with torch.no_grad():
                self.model.swiftkv(False)
                self.model.eval()
                teacher_outputs = self.model(
                    **batch,
                    output_hidden_states=(self.config.decoder_loss_mult > 0),
                )

            self.model.swiftkv(True)
            self.model.train()
            student_outputs = self.model(
                **batch,
                output_hidden_states=(self.config.decoder_loss_mult > 0),
            )

            distill_loss = self.distillation_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                temperature=self.config.temperature,
            )

            logits = student_outputs.logits

            see_memory_usage(f"{sub_step_id=} after forward", force=False)

            #print_rank(f"{labels=}", skip=False)
            #print_rank(f"{logits=}", skip=False)
            # print_rank(f"logit nans: {torch.isnan(logits).sum()}", skip=False)
            # print_rank(f"logit infs: {torch.isinf(logits).sum()}", skip=False)


            if all((shift_labels == -100).squeeze()):
                # this is the case where all labels in the micro-batch are -100 (very common for SFT) - CE returns `nan` in this case, so we don't want to call loss and instead create a differentiable loss `0` which will also set all the grads to `0` in `backward` - the effect of this is akin to a perfect score where the model needs no adjustment since grads will be all zeros.
                loss = (logits.sum() * 0.0).float()
                #loss = FakeLoss.apply(logits)
            else:
                loss = self.distillation_loss(
                    student_outputs.logits,
                    teacher_outputs.logits,
                    temperature=self.config.temperature,
                )

            #loss = outputs.loss
            print_rank(f"LOSS local {loss=}", skip=False)

            # free up temp mem (e.g. outputs.logits are huge)
            del teacher_outputs
            del student_outputs

            see_memory_usage(f"{sub_step_id=} before gathered loss", force=False)
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

            # since each shard may have a variable number of skipped elemented - need to calculate a weighted mean depending on each rank's contribution - this will also take care of loss=0 when all elements are -100 in a shard
            # XXX: not expecting a total of 0-non-skipped items for div
            loss = sum(losses_per_rank[rank] * non_skipped_items[rank] for rank in range(sp_world_size)) / sum(non_skipped_items.values())
            # this is a much simpler version w/o weighting
            # skip 0.0 entries when calculating total loss per batch
            # loss = torch.stack(list(l for l in losses_per_rank if l != 0)).mean()

            #loss = torch.cat([l.unsqueeze() for l in losses_per_rank], dim=0).mean()
            #loss = sum(loss_per_rank) # / sp_world_size
            #loss = sum(tensor_list)
            #print_rank(f"LOSS averaged {loss=}", skip=False)
            #print("LOSS", loss)
            see_memory_usage(f"{sub_step_id=} after gathered loss", force=False)

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
            # see_memory_usage(f"{sub_step_id=} after cat", force=False)

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

            see_memory_usage(f"{sub_step_id=} before backward", force=False)
            #import gc; gc.collect()
            self.model.backward(loss)


            # print_rank(f"{labels[0][70:80]=}", skip=False)
            # print_rank(f"{logits[0][70:80]=}", skip=False)
            # print_rank(f'{batch["input_ids"][0][70:80]=}', skip=False)
            # print_rank(f'{batch["input_ids"].grad[0][70:80]=}', skip=False)
            # print_rank(f"{logits.grad[0][70:80]=}", skip=False)
            # exit()

            print_rank0(f"zero loss: {loss}", skip=False)
            # print_rank0(f"zero loss: {avg_loss}", skip=False)
            see_memory_usage(f"{sub_step_id=} after backward", force=False)

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

    def distillation_loss(
        self, student_output, teacher_output, temperature=1.0, dim=-1
    ):
        # Soften the student logits by applying softmax first and log() second
        soft_targets = F.softmax(teacher_output / temperature, dim=dim)
        soft_prob = F.log_softmax(student_output / temperature, dim=dim)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the
        # authors of the paper "Distilling the knowledge in a neural network"
        return torch.mean(
            torch.sum(
                soft_targets * (soft_targets.log() - soft_prob),
                dim=dim,
            )
            * temperature**2
        )
