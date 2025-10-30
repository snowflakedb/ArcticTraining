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

from arctic_training.debug.utils import pr0
from arctic_training.logging import logger
from arctic_training.model.factory import ModelFactory


class HFModelFactory(ModelFactory):
    name = "huggingface"

    def create_config(self):
        return AutoConfig.from_pretrained(self.config.name_or_path)

    def create_model(self, model_config) -> PreTrainedModel:

        # XXX: temp - using a local copy of the HF modeling code
        config = self.create_config()

        if config.architectures[0] == "Qwen3MoeForCausalLM":
            pr0("Using custom Qwen3MoeForCausalLM", force=True)
            from arctic_training.model.qwen3_moe import Qwen3MoeForCausalLM

            return Qwen3MoeForCausalLM.from_pretrained(
                self.config.name_or_path,
                config=model_config,
                attn_implementation=self.config.attn_implementation,
                dtype=self.config.dtype.value,
            )
        elif config.architectures[0] == "GptOssForCausalLM" and not str(self.config.name_or_path).startswith(
            "openai/gpt-oss-"
        ):
            # for some reason if we are using a copy of GptOssForCausalLM the official gpt-oss models with Mxfp4 weights leave the model on a meta device, but it works fine if we use transformers.models.gpt_oss.modeling_gpt_oss.GptOssForCausalLM which is identical.
            # it fails then when trying to copy the weights: NotImplementedError: Cannot copy out of meta tensor; no data!
            # if we use for example unsloth/gpt-oss-20b-BF16 the local copy works fine.
            # so for now we will use a local copy only for non-openai/gpt-oss-* models.

            pr0("Using custom GptOssForCausalLM", force=True)

            from arctic_training.model.gpt_oss import GptOssForCausalLM

            # a failed attempt to make the local modeling code copy work with official mxfp4 models
            # # https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers
            # import transformers.models.gpt_oss.modeling_gpt_oss
            # transformers.models.gpt_oss.modeling_gpt_oss.GptOssForCausalLM = GptOssForCausalLM
            # import torch
            # from transformers import Mxfp4Config
            # quantization_config = Mxfp4Config(**config.quantization_config)
            # print(quantization_config)
            # quantization_config = Mxfp4Config(dequantize=True)
            # model_kwargs = dict(
            #     # attn_implementation="eager",
            #     dtype=torch.bfloat16,
            #     quantization_config=quantization_config,
            #     use_cache=False,
            #     #device_map="auto",
            # )
            # model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", **model_kwargs)
            # return model

            return GptOssForCausalLM.from_pretrained(
                self.config.name_or_path,
                config=model_config,
                attn_implementation=self.config.attn_implementation,
                dtype=self.config.dtype.value,
            )

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
