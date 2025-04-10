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

from typing import Any, Union

import torch
import torch.nn.functional as F
from arctic_training import (
    HFCheckpointEngine,
    HFModelFactory,
    ModelConfig,
    SFTTrainer,
    TrainerConfig,
    logger,
)
from arctic_training.trainer.sft_trainer import to_device
from deepspeed.runtime.zero import GatheredParameters
from projects.swiftkv import llama_swiftkv, qwen2_swiftkv

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
        if model_type == "llama":
            hf_config = llama_swiftkv.LlamaSwiftKVConfig.from_dict(config_dict)
        elif model_type == "qwen2":
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
            with GatheredParameters(layer.parameters(), modifier_rank=0):
                q_proj_swiftkv.weight.data.copy_(layer.self_attn.q_proj.weight.data)
                q_proj_swiftkv.weight.requires_grad = True
                if getattr(q_proj_swiftkv, "bias", None) is not None:
                    q_proj_swiftkv.bias.data.copy_(layer.self_attn.q_proj.bias.data)
                    q_proj_swiftkv.bias.requires_grad = True
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
                kv_proj_swiftkv = getattr(this_attn, f"{param}_swiftkv")
                # Initialize k_proj or v_proj weights
                weights = [kv_proj_swiftkv.weight] + [
                    getattr(attn, f"{param}").weight for attn in next_attn
                ]
                with GatheredParameters(weights, modifier_rank=0):
                    weights[0].data.copy_(
                        sum(weights[1:]) / model.config.key_value_group_size
                    )
                    kv_proj_swiftkv.weight.requires_grad = True
                # Initialize k_proj or v_proj biases (if they exist)
                if getattr(kv_proj_swiftkv, "bias", None) is not None:
                    biases = [kv_proj_swiftkv.bias] + [
                        getattr(attn, f"{param}").bias for attn in next_attn
                    ]
                    with GatheredParameters(biases, modifier_rank=0):
                        biases[0].data.copy_(
                            sum(biases[1:]) / model.config.key_value_group_size
                        )
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

    def loss(self, batch: Any) -> torch.Tensor:
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
