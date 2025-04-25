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

from peft import get_peft_model
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel

from arctic_training.logging import logger
from arctic_training.model.factory import ModelFactory


class HFModelFactory(ModelFactory):
    name = "huggingface"

    def create_config(self):
        return AutoConfig.from_pretrained(self.config.name_or_path)

    def create_model(self, model_config) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(
            self.config.name_or_path,
            config=model_config,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=self.config.dtype.value,
        )

    @staticmethod
    def make_model_gradient_checkpointing_compatible(
        model: PreTrainedModel,
    ) -> PreTrainedModel:
        # Higgingface added this enable input require grads function to make gradient checkpointing work for lora-only optimization
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        return model

    def post_create_model_callback(self, model):
        if self.config.peft_config is not None:
            model = get_peft_model(model, self.config.peft_config_obj)
            trainable_params, all_params = model.get_nb_trainable_parameters()
            logger.info(
                f"Applied PEFT config to model: Total params: {all_params}, Trainable"
                f" params: {trainable_params} ({100*trainable_params/all_params:.2f}%)"
            )

        if not self.config.disable_activation_checkpoint:
            model.gradient_checkpointing_enable()
            model = self.make_model_gradient_checkpointing_compatible(model)

        return model
