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
from typing import Any, Union

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
from arctic_training.trainer.trainer import ChunkedMemEfficientLoss, UlyssesAttentionHFFwdLossBwdWithLogits


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

    def sp_fwd_loss_bwd(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)

        ulysses = SwiftKVUlyssesAttentionHFFwdLossBwdWithLogits(
            model=self.model,
            model_unwrapped=self.model_unwrapped,
            device=self.device,
            num_loss_logit_shards="auto",
        )
        return ulysses.sp_fwd_loss_bwd(batch)

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


class SwiftKVUlyssesAttentionHFFwdLossBwdWithLogits(UlyssesAttentionHFFwdLossBwdWithLogits):

    def __init__(self,
                 model,
                 model_unwrapped,
                 device,
                 num_loss_logit_shards="auto",
                 temperature: float = 1.0,
                 **kwargs,
        ):
        super().__init__(model, model_unwrapped, device, num_loss_logit_shards, **kwargs)
        self.temperature = temperature

    def forward(self, batch):
        with torch.no_grad():
            self.model.swiftkv(False)
            self.model.eval()
            teacher_outputs = self.model(
                **batch,
                output_hidden_states=False,
            )

        self.model.swiftkv(True)
        self.model.train()
        student_outputs = self.model(
            **batch,
            output_hidden_states=False,
        )

        self.logits = torch.cat([student_outputs.logits, teacher_outputs.logits], dim=-1)

        return student_outputs

    def distillation_loss(
        self,
        logits,
        labels=None,
        vocab_size=None,
        shift_labels=None,
    ):
        student_logits, teacher_logits = torch.split(logits, logits.shape[-1] // 2, dim=-1)

        # Soften the student logits by applying softmax first and log() second
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the
        # authors of the paper "Distilling the knowledge in a neural network"
        return torch.mean(
            torch.sum(
                soft_targets * (soft_targets.log() - soft_prob),
                dim=-1,
            )
            * self.temperature**2
        )

    def compute_loss(self, labels, shift_labels):
        if all((shift_labels == -100).squeeze()):
            # this is the case where all labels in a micro-batch are -100 (very common for SFT) - CE returns `nan` in this case, so we don't want to call loss and instead create a differentiable loss `0` which will also set all the grads to `0` in `backward` - the effect of this is akin to a perfect score where the model needs no adjustment since grads will be all zeros.
            # XXX: should this be float and not the original dtype?
            loss = (self.logits.sum() * 0.0).float()
        else:
            if self.num_loss_logit_shards == "auto":
                # parameterize to about 1GB fp32 logits shards
                slice_size_in_gb = 1 # XXX: make configurable?
                size_in_gb = self.logits.numel() * 4 / 2**30 # fp32
                # the sp shard's seqlen sp shard can be easily not divisible by the derived number of chunked loss shards, so we use the uppper ceiling and allow the last chunk to be shorter than the rest
                self.num_loss_logit_shards = math.ceil(size_in_gb / slice_size_in_gb)
                #print(f"derived {self.num_loss_logit_shards} shards for size {size_in_gb}GB")
            if self.num_loss_logit_shards > 1:
                loss = ChunkedMemEfficientLoss.apply(
                    self.distillation_loss,
                    self.logits,
                    None,
                    shift_labels,
                    self.num_loss_logit_shards,
                )
            else:
                # XXX: for some reason this fails with zero1
                loss = self.distillation_loss(
                    logits=self.logits,
                    labels=None,
                    vocab_size=None,
                    shift_labels=shift_labels,
                )

        self.loss = loss
        return loss