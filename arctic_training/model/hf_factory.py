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

import deepspeed
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel

from arctic_training.logging import logger
from arctic_training.model.factory import ModelFactory


class HFModelFactory(ModelFactory):
    name = "huggingface"

    def create_config(self):
        config = AutoConfig.from_pretrained(self.config.name_or_path)

        # override hf model config if we have some custom config
        for k, v in self.config.hf_config_kwargs.items():
            setattr(config, k, v)

        return config

    def create_model(self, model_config) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(
            self.config.name_or_path,
            config=model_config,
            attn_implementation=self.config.attn_implementation,
            dtype=self.config.dtype.value,
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
        if self.config.fp8_recipe is not None:
            import transformer_engine.pytorch as te

            replace_module_count = 0

            def replace_linears(module):
                for name, child in module.named_children():
                    if any(tm in name for tm in self.config.fp8_target_modules):
                        if not isinstance(child, nn.Linear):
                            logger.warning(f"Module {name} matched for FP8 conversion but is not nn.Linear, skipping.")
                        else:
                            te_linear = te.Linear(
                                in_features=child.in_features,
                                out_features=child.out_features,
                                bias=child.bias is not None,
                                device=child.weight.device,
                            )

                            if hasattr(child.weight, "ds_id"):
                                # Parameter is managed by DeepSpeed Zero-3
                                params_to_fetch = [child.weight]
                                if child.bias is not None:
                                    params_to_fetch.append(child.bias)

                                with deepspeed.zero.GatheredParameters(params_to_fetch, modifier_rank=None):
                                    te_linear.weight.data.copy_(child.weight.data)
                                    if child.bias is not None:
                                        te_linear.bias.data.copy_(child.bias.data)
                            else:
                                # Regular parameter, copy directly
                                te_linear.weight.data.copy_(child.weight.data)
                                if child.bias is not None:
                                    te_linear.bias.data.copy_(child.bias.data)

                            setattr(module, name, te_linear)

                            nonlocal replace_module_count
                            replace_module_count += 1

                    replace_linears(child)

            replace_linears(model)
            if replace_module_count == 0:
                raise ValueError(
                    "FP8 recipe specified but no modules were replaced. Please check `fp8_target_modules`."
                )

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
