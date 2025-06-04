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
from typing import Any, List
from typing import Union

import torch
import torch.nn.functional as F
from deepspeed.runtime.zero import GatheredParameters

from arctic_training import HFCheckpointEngine
from arctic_training import HFModelFactory
from arctic_training import ModelConfig
from arctic_training import SFTTrainer
from arctic_training import TrainerConfig
from arctic_training import logger
#from arctic_training.deepspeed import ChunkedMemEfficientLoss
from arctic_training.trainer.sft_trainer import to_device
from projects.swiftkv.models import DeepseekV2SwiftKVConfig
from projects.swiftkv.models import LlamaSwiftKVConfig
from projects.swiftkv.models import Qwen2SwiftKVConfig
from projects.swiftkv.models import register_all_swiftkv
from projects.swiftkv.models.deepseek_v2 import register_deepseek_v2

register_all_swiftkv()
register_deepseek_v2()  # Explicitly register because it's not in transformers


class SwiftKVModelConfig(ModelConfig):
    num_key_value_layers: int
    key_value_group_size: int = 1


class SwiftKVModelFactory(HFModelFactory):
    name = "swiftkv"
    config: SwiftKVModelConfig

    def post_create_config_callback(self, hf_config):
        config_dict = hf_config.to_dict()

        model_type = config_dict.get("model_type")
        if model_type in ["deepseek_v2", "deepseek_v2_swiftkv"]:
            hf_config = DeepseekV2SwiftKVConfig.from_dict(config_dict)
        elif model_type in ["llama", "llama_swiftkv"]:
            hf_config = LlamaSwiftKVConfig.from_dict(config_dict)
        elif model_type in ["qwen2", "qwen2_swiftkv"]:
            hf_config = Qwen2SwiftKVConfig.from_dict(config_dict)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        hf_config.num_key_value_layers = self.config.num_key_value_layers
        hf_config.key_value_group_size = self.config.key_value_group_size

        return hf_config

    def post_create_model_callback(self, model):
        if model.config.model_type == "deepseek_v2":
            if model.config.q_lora_rank is None:
                q_modules = ["q_proj"]
            else:
                q_modules = ["q_a_proj", "q_b_proj", "q_a_layernorm"]
            kv_modules = ["kv_a_proj_with_mqa", "kv_b_proj", "kv_a_layernorm"]
        else:
            q_modules = ["q_proj"]
            kv_modules = ["k_proj", "v_proj"]

        # Freeze all teacher parameters
        for param in model.parameters():
            param.requires_grad = False

        # Initialize the swiftkv norm to the norm of the first non-kv layer.
        layer = model.model.layers[model.config.num_key_value_layers]
        with GatheredParameters(
            list(model.model.norm_swiftkv.parameters()) + list(layer.input_layernorm.parameters()), modifier_rank=0
        ):
            model.model.norm_swiftkv.weight.data.copy_(layer.input_layernorm.weight.data)
        model.model.norm_swiftkv.weight.requires_grad = True

        # Initialize all query parameters directly from the corresponding teacher layer.
        for layer in model.model.layers[model.config.num_key_value_layers :]:
            attn = layer.self_attn
            with GatheredParameters(attn.parameters(), modifier_rank=0):
                for q_module in q_modules:
                    teacher_params = getattr(attn, q_module).parameters()
                    student_params = getattr(attn, f"{q_module}_swiftkv").parameters()
                    for teacher_param, student_param in zip(teacher_params, student_params):
                        student_param.data.copy_(teacher_param.data)
                        student_param.requires_grad = True

        # Initialize all kv parameters to the mean of the teacher layers in each kv group.
        for idx, layer in enumerate(model.model.layers[model.config.num_key_value_layers :]):
            attn = layer.self_attn
            if idx % model.config.key_value_group_size == 0:
                # This layer has swiftkv parameters, zero them out.
                kv_attn = attn
                with GatheredParameters(kv_attn.parameters(), modifier_rank=0):
                    # Zero out the swiftkv parameters
                    for kv_module in kv_modules:
                        for param in getattr(kv_attn, f"{kv_module}_swiftkv").parameters():
                            param.data.zero_()
                            param.requires_grad = True
            with GatheredParameters(attn.parameters(), modifier_rank=0):
                # Accumulate the teacher parameters into the swiftkv parameters.
                for kv_module in kv_modules:
                    teacher_params = getattr(attn, kv_module).parameters()
                    student_params = getattr(kv_attn, f"{kv_module}_swiftkv").parameters()
                    for teacher_param, student_param in zip(teacher_params, student_params):
                        student_param.data.add_(teacher_param.data / model.config.key_value_group_size)

        model.gradient_checkpointing_enable()
        return model


class SwiftKVTrainerConfig(TrainerConfig):
    logits_loss_temp: float = 2.0
    hidden_loss_mult: float = 1.0
    hidden_loss_layer: int = -1


class SwiftKVTrainer(SFTTrainer):
    name = "swiftkv"
    config: SwiftKVTrainerConfig
    model_factory: SwiftKVModelFactory
    checkpoint_engine: Union[HFCheckpointEngine]

    def forward(self, batch):
        batch = to_device(batch, self.device)

        with torch.no_grad():
            self.model.swiftkv(False)
            self.model.eval()
            teacher_outputs = self.model(**batch, output_hidden_states=True)

        self.model.swiftkv(True)
        self.model.train()
        student_outputs = self.model(**batch, output_hidden_states=True)

        return student_outputs, teacher_outputs

    def loss(self, batch) -> torch.Tensor:
        import torch
        batch = to_device(batch, self.device)

        if self.config.sequence_parallel_size == 1:
            student_outputs, teacher_outputs = self.forward(batch)

            logits_loss = self.logits_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                temperature=self.config.logits_loss_temp,
                mask=(batch["labels"] != -100),
            )

            hidden_loss = self.hidden_loss(
                student_outputs.hidden_states[self.config.hidden_loss_layer],
                teacher_outputs.hidden_states[self.config.hidden_loss_layer],
            )

            logger.info(
                f"student loss: {student_outputs.loss.item()}, "
                f"teacher loss: {teacher_outputs.loss.item()}, "
                f"logits loss: {logits_loss.item()}, "
                f"hidden loss: {hidden_loss.item()}"
            )

            loss = logits_loss + self.config.hidden_loss_mult * hidden_loss
        else:
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
            student_outputs, teacher_outputs = self.forward(batch)
            logits = torch.cat([
                student_outputs.logits,
                student_outputs.hidden_states[self.config.hidden_loss_layer],
                teacher_outputs.logits,
                teacher_outputs.hidden_states[self.config.hidden_loss_layer],
            ], dim=-1)
            del student_outputs.hidden_states
            del teacher_outputs.hidden_states

            # XXX: parameterize
            num_loss_logit_shards: Any = "auto"

            if False:  # all((shift_labels == -100).squeeze()):
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
                    loss = ChunkedMemEfficientLoss.apply(
                        self.sp_loss,
                        logits,
                        self.model_unwrapped.config.vocab_size,
                        shift_labels,
                        num_loss_logit_shards,
                    )
                else:
                    # XXX: for some reason this was failing with zero1 w/ previous design - need to retest with the new design
                    loss = self.sp_loss(
                        logits=logits,
                        labels=None,
                        vocab_size=self.model_unwrapped.config.vocab_size,
                        shift_labels=shift_labels,
                    )

            # differentiable weighted per-shard-loss aggregation across ranks
            import torch.distributed.nn.functional

            losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=self.sp_group)
            #good_tokens = sum((shift_labels != -100).view(-1))
            good_tokens = sum((shift_labels != -1000).view(-1))
            good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=self.sp_group)
            loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(self.sp_world_size)) / sum(
                good_tokens_per_rank
            )

        return loss

    def logits_loss(self, student_output, teacher_output, temperature=1.0, dim=-1, mask=None):
        # Soften the student logits by applying softmax first and log() second
        soft_targets = F.softmax(teacher_output / temperature, dim=dim)
        soft_prob = F.log_softmax(student_output / temperature, dim=dim)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the
        # authors of the paper "Distilling the knowledge in a neural network"
        loss = torch.sum(
            soft_targets * (soft_targets.log() - soft_prob),
            dim=dim,
        )
        if mask is not None:
            loss = loss * mask
        return torch.mean(loss * temperature**2)

    def hidden_loss(self, student_hidden_states, teacher_hidden_states):
        return F.mse_loss(student_hidden_states, teacher_hidden_states)

    def distillation_loss_sp(self, logits, labels=None, vocab_size=None, shift_labels=None):
        student_logits, teacher_logits = torch.chunk(logits, 2, dim=-1)

        return self.logits_loss(
            student_logits,
            teacher_logits,
            temperature=self.config.logits_loss_temp,
            mask=(shift_labels != -100),
        )

    def sp_loss(self, logits, labels=None, vocab_size=None, shift_labels=None):
        vocab_size = self.model_unwrapped.config.vocab_size
        hidden_size = self.model_unwrapped.config.hidden_size
        student_logits, student_hidden, teacher_logits, teacher_hidden = torch.split(
            logits, [vocab_size, hidden_size, vocab_size, hidden_size], dim=-1
        )
        # logits loss
        logits_loss = self.logits_loss(
            student_logits,
            teacher_logits,
            temperature=self.config.logits_loss_temp,
            mask=(shift_labels != -100),
        )
        # hidden loss
        hidden_loss = self.hidden_loss(
            student_hidden,
            teacher_hidden,
        )
        return logits_loss + self.config.hidden_loss_mult * hidden_loss


class ChunkedMemEfficientLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss_fn, logits, vocab_size, shift_labels, shards) -> torch.Tensor:
        """
        logits doesn't have to be divisible by shards, the last shard will be shorter than the rest.
        """
        ctx.save_for_backward(logits, shift_labels)
        ctx.loss_fn = loss_fn
        ctx.vocab_size = vocab_size
        ctx.shards = shards

        with torch.no_grad():
            seqlen = shift_labels.shape[1]
            shard_step = math.ceil(seqlen / shards)
            loss_shards = []
            total_good_items = 0

            # since -100s are ignored we have to perform a weighted average on each loss slice as each slice may contribute a different number of non- -100 labels
            # if seqlen / shards != 0 - the last chunk is just shorter than the rest but no data is ignored
            for i in range(shards):
                # XXX: here and everywhere don't make a copy, pass the slice or perhaps narrow/view?
                shift_labels_shard = shift_labels[:, i * shard_step : (i + 1) * shard_step]
                #if all((shift_labels_shard == -100).squeeze()):
                #    continue  # ignore this shard
                loss_shard = loss_fn(
                    logits=logits[:, i * shard_step : (i + 1) * shard_step, :],
                    labels=None,
                    vocab_size=vocab_size,
                    shift_labels=shift_labels_shard,
                )
                good_items = sum((shift_labels_shard != -1000).squeeze())
                loss_shards.append(loss_shard * good_items)
                total_good_items += good_items
            total_loss = torch.cat([l.unsqueeze(0) for l in loss_shards], dim=0).sum()
            weighted_loss = total_loss / total_good_items

        # weighted_loss.requires_grad = True
        return weighted_loss

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:

        logits, shift_labels = ctx.saved_tensors
        loss_fn = ctx.loss_fn
        vocab_size = ctx.vocab_size
        shards = ctx.shards

        grad = grads[0]
        logits_grad = torch.zeros_like(logits)
        # logits_grad = torch.zeros(logits.shape, device=logits.device, dtype=grad.dtype, requires_grad=logits.requires_grad)

        logits_shards = list(torch.chunk(logits, chunks=shards, dim=1))
        shift_labels_shards = list(torch.chunk(shift_labels, chunks=shards, dim=1))
        del logits
        del shift_labels
        ctx.logits = None
        ctx.shift_labels = None
        ctx.loss_fn = None
        ctx.vocab_size = None
        ctx.shards = None

        # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
        shard_step = logits_shards[0].numel()
        for i in range(shards):
            logits_shard = logits_shards.pop(0)
            shift_labels_shard = shift_labels_shards.pop(0)

            shard_offset = i * shard_step
            # this will enable gradual population of the pre-allocated `logits_shard.grad` during `torch.autograd.backward` calls
            logits_shard.grad = (
                logits_grad.view(-1).narrow(0, shard_offset, logits_shard.numel()).view_as(logits_shard)
            )

            with torch.enable_grad():
                if False:  # all((shift_labels_shard == -100).squeeze()):
                    # fake loss calculation, since CE will return nan, but grads will be set
                    # a normal loss_fn upcasts logits to float so match it
                    loss_shard = (logits_shard.sum() * 0.0).float()
                else:
                    loss_shard = loss_fn(
                        logits=logits_shard.requires_grad_(),
                        labels=None,
                        vocab_size=vocab_size,
                        shift_labels=shift_labels_shard,
                    )

            torch.autograd.backward(loss_shard, grad)

        logits_grad /= shards

        # print(f"returning {logits_grad.norm()=}")
        # print(f"returning {logits_grad=}")
        # only logits (2nd arg) needs grads
        return None, logits_grad, None, None, None