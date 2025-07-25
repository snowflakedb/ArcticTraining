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
import torch.nn.functional as F
from deepspeed.runtime.sequence_parallel.ulysses_sp import sequence_tiled_compute
from deepspeed.runtime.zero import GatheredParameters
from torch.distributed import ReduceOp

from arctic_training import HFCheckpointEngine
from arctic_training import HFModelFactory
from arctic_training import ModelConfig
from arctic_training import SFTTrainer
from arctic_training import TrainerConfig
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
    """
    Initial number of layers that compute KV cache. The output from layer
    `num_key_value_layers` is used to compute the KV for all subsequent layers.
    """

    key_value_group_size: int = 1
    """
    Number of consecutive layers that share the same KV cache, only applies to
    layers after `num_key_value_layers`.
    """


class SwiftKVTrainerConfig(TrainerConfig):
    logits_loss_temp: float = 2.0
    """Temperature for the distillation (KL-div) loss on logits."""

    hidden_loss_mult: float = 1.0
    """
    Weight for the distillation (MSE) loss on hidden states. The final loss
    is computed as `logits_loss + hidden_loss_mult * hidden_loss`.
    """

    hidden_loss_layer: int = -2
    """The index of the layer whose output is used for the hidden loss."""


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
        if model.config.model_type in ["deepseek_v2", "deepseek_v2_swiftkv"]:
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

        student_outputs, teacher_outputs = self.forward(batch)

        student_logits = student_outputs.logits
        student_hidden = student_outputs.hidden_states[self.config.hidden_loss_layer]
        teacher_logits = teacher_outputs.logits
        teacher_hidden = teacher_outputs.hidden_states[self.config.hidden_loss_layer]

        def loss_fn(student_concat, teacher_logits, teacher_hidden, mask):
            student_logits, student_hidden = student_concat.split(
                [teacher_logits.size(-1), teacher_hidden.size(-1)], dim=-1
            )

            # 1. Logits distillation loss for non-masked positions.
            # Soften the student logits by applying softmax first and log() second
            soft_targets = F.softmax(teacher_logits / self.config.logits_loss_temp, dim=-1)
            soft_prob = F.log_softmax(student_logits / self.config.logits_loss_temp, dim=-1)

            # Calculate the soft logits loss. Scaled by T**2 as suggested by the
            # authors of the paper "Distilling the knowledge in a neural network"
            logits_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob), dim=-1)
            logits_loss = logits_loss * mask  # Zero out the masked positions
            logits_loss = torch.mean(logits_loss * self.config.logits_loss_temp**2)

            if self.config.hidden_loss_mult > 0:
                # 2. Hidden states MSE loss for all masked and non-masked positions.
                hidden_loss = F.mse_loss(student_hidden, teacher_hidden)
                return logits_loss + self.config.hidden_loss_mult * hidden_loss
            else:
                return logits_loss

        if self.config.sequence_parallel_size > 1:
            shift_labels = batch["shift_labels"]
            batch_size, seqlen = shift_labels.shape
            num_shards = self.get_num_shards(batch_size, seqlen)
            kwargs_to_shard = {
                "student_concat": torch.cat([student_logits, student_hidden], dim=-1),
                "teacher_logits": teacher_logits,
                "teacher_hidden": teacher_hidden,
                "mask": shift_labels != -100,
            }

            loss = sequence_tiled_compute(
                loss_fn,
                seqlen,
                num_shards,
                kwargs_to_shard,
                kwargs_to_pass={},
                grad_requiring_tensor_key="student_concat",
                compute_params=[],
                output_unshard_dimension=0,  # loss is a scalar
            )

            # Average the loss across all sequence parallel ranks using a differentiable all_reduce.
            loss = torch.distributed.nn.functional.all_reduce(loss, op=ReduceOp.AVG, group=self.sp_group)

            return loss
        else:
            student_concat = torch.cat([student_logits, student_hidden], dim=-1)
            return loss_fn(
                student_concat=student_concat,
                teacher_logits=teacher_logits,
                teacher_hidden=teacher_hidden,
                mask=(batch["labels"] != -100),
            )

    def get_num_shards(self, batch_size, seqlen):
        slice_size_in_gb = 1  # XXX: make configurable?
        vocab_size = self.model_unwrapped.config.vocab_size
        logits_numel = batch_size * seqlen * vocab_size
        size_in_gb = logits_numel * 4 / 2**30  # fp32
        # the sp shard's seqlen sp shard can be easily not divisible by the derived number
        # of chunked loss shards, so we use the uppper ceiling and allow the last chunk to
        # be shorter than the rest
        return math.ceil(size_in_gb / slice_size_in_gb)
