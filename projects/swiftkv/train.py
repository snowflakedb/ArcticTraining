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
        if model_type == "deepseek_v2":
            hf_config = DeepseekV2SwiftKVConfig.from_dict(config_dict)
        elif model_type == "llama":
            hf_config = LlamaSwiftKVConfig.from_dict(config_dict)
        elif model_type == "qwen2":
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

        return loss

    def distillation_loss(self, student_output, teacher_output, temperature=1.0, dim=-1):
        # Soften the student logits by applying softmax() first and log() second
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
