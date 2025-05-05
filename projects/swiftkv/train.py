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
import torch.nn.functional as F
from deepspeed.runtime.zero import GatheredParameters

from arctic_training import HFCheckpointEngine
from arctic_training import HFModelFactory
from arctic_training import ModelConfig
from arctic_training import SFTTrainer
from arctic_training import TrainerConfig
from arctic_training import logger
from arctic_training.deepspeed import ChunkedMemEfficientLoss
from arctic_training.trainer.sft_trainer import to_device
from projects.swiftkv import llama_swiftkv
from projects.swiftkv import qwen2_swiftkv

llama_swiftkv.register_auto()
qwen2_swiftkv.register_auto()


class SwiftKVModelConfig(ModelConfig):
    num_key_value_layers: int
    key_value_group_size: int = 1


class SwiftKVModelFactory(HFModelFactory):
    name = "swiftkv"
    config: SwiftKVModelConfig

    def post_create_config_callback(self, hf_config):
        config_dict = hf_config.to_dict()

        model_type = config_dict.get("model_type")
        if model_type in ["llama", "llama_swiftkv"]:
            hf_config = llama_swiftkv.LlamaSwiftKVConfig.from_dict(config_dict)
        elif model_type in ["qwen2", "qwen2_swiftkv"]:
            hf_config = qwen2_swiftkv.Qwen2SwiftKVConfig.from_dict(config_dict)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        hf_config.num_key_value_layers = self.config.num_key_value_layers
        hf_config.key_value_group_size = self.config.key_value_group_size

        return hf_config

    def post_create_model_callback(self, model):
        # Freeze all teacher parameters
        for param in model.parameters():
            param.requires_grad = False

        # Initialize student layers
        model.model.norm_swiftkv.weight.requires_grad = True
        for layer in model.model.layers[model.config.num_key_value_layers :]:
            # Initialize q_proj_swiftkv
            q_proj_swiftkv = layer.self_attn.q_proj_swiftkv
            if not model.config.swiftkv:
                with GatheredParameters(layer.parameters(), modifier_rank=0):
                    q_proj_swiftkv.weight.data.copy_(layer.self_attn.q_proj.weight.data)
                    if getattr(q_proj_swiftkv, "bias", None) is not None:
                        q_proj_swiftkv.bias.data.copy_(layer.self_attn.q_proj.bias.data)
            q_proj_swiftkv.weight.requires_grad = True
            if getattr(q_proj_swiftkv, "bias", None) is not None:
                q_proj_swiftkv.bias.requires_grad = True
        for layer_idx in range(
            model.config.num_key_value_layers,
            model.config.num_hidden_layers,
            model.config.key_value_group_size,
        ):
            this_attn = model.model.layers[layer_idx].self_attn
            next_attn = [model.model.layers[layer_idx + i].self_attn for i in range(model.config.key_value_group_size)]
            for param in ("k_proj", "v_proj"):
                kv_proj_swiftkv = getattr(this_attn, f"{param}_swiftkv")
                # Initialize k_proj or v_proj weights
                if not model.config.swiftkv:
                    weights = [kv_proj_swiftkv.weight] + [getattr(attn, f"{param}").weight for attn in next_attn]
                    with GatheredParameters(weights, modifier_rank=0):
                        weights[0].data.copy_(sum(weights[1:]) / model.config.key_value_group_size)
                    # Initialize k_proj or v_proj biases (if they exist)
                    if getattr(kv_proj_swiftkv, "bias", None) is not None:
                        biases = [kv_proj_swiftkv.bias] + [getattr(attn, f"{param}").bias for attn in next_attn]
                        with GatheredParameters(biases, modifier_rank=0):
                            biases[0].data.copy_(sum(biases[1:]) / model.config.key_value_group_size)
                kv_proj_swiftkv.weight.requires_grad = True
                if getattr(kv_proj_swiftkv, "bias", None) is not None:
                    kv_proj_swiftkv.bias.requires_grad = True
        model.gradient_checkpointing_enable()
        return model


class SwiftKVTrainerConfig(TrainerConfig):
    temperature: float = 1.0


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
            teacher_outputs = self.model(**batch)

        self.model.swiftkv(True)
        self.model.train()
        student_outputs = self.model(**batch)

        return student_outputs, teacher_outputs

    def loss(self, batch) -> torch.Tensor:
        import torch
        batch = to_device(batch, self.device)

        if self.config.sequence_parallel_size == 1:
            student_outputs, teacher_outputs = self.forward(batch)

            loss = self.distillation_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                temperature=self.config.temperature,
                mask=(batch["labels"] != -100),
            )

            logger.info(
                f"student loss: {student_outputs.loss.item()}, teacher loss:"
                f" {teacher_outputs.loss.item()}, distill loss: {loss.item()}"
            )
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
            logits = torch.cat([student_outputs.logits, teacher_outputs.logits], dim=-1)

            # XXX: parameterize
            num_loss_logit_shards: Any = "auto"

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
                    loss = ChunkedMemEfficientLoss.apply(
                        self.distillation_loss_sp,
                        logits,
                        self.model_unwrapped.config.vocab_size,
                        shift_labels,
                        num_loss_logit_shards,
                    )
                else:
                    # XXX: for some reason this was failing with zero1 w/ previous design - need to retest with the new design
                    loss = self.distillation_loss_sp(
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

    def loss_old(self, batch: Any) -> torch.Tensor:
        batch = to_device(batch, self.device)

        with torch.no_grad():
            self.model.swiftkv(False)
            self.model.eval()
            teacher_outputs = self.model(**batch)

        self.model.swiftkv(True)
        self.model.train()
        student_outputs = self.model(**batch)

        loss = self.distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            temperature=self.config.temperature,
        )

        logger.info(
            f"student loss: {student_outputs.loss.item()}, teacher loss:"
            f" {teacher_outputs.loss.item()}, distill loss: {loss.item()}"
        )

        torch.cuda.synchronize()

        return loss

    def distillation_loss(self, student_output, teacher_output, temperature=1.0, dim=-1, mask=None):
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

    def distillation_loss_sp(self, logits, labels=None, vocab_size=None, shift_labels=None):
        student_logits, teacher_logits = torch.chunk(logits, 2, dim=-1)

        return self.distillation_loss(
            student_logits,
            teacher_logits,
            temperature=self.config.temperature,
            mask=(shift_labels != -100),
        )
