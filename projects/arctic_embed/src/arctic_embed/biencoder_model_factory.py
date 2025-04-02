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
from typing import Dict
from typing import cast

from peft import get_peft_model
from peft.config import PeftConfig
from transformers import AutoConfig
from transformers import AutoModel

from arctic_training.config.model import ModelConfig
from arctic_training.model.factory import ModelFactory
from arctic_training.model.hf_factory import HFModelFactory

from .core.biencoder_model import Biencoder
from .core.biencoder_model import PoolingOption


class BiencoderModelConfig(ModelConfig):
    type: str = "biencoder"
    pooling: PoolingOption = "first_token"
    kwargs: Dict[str, Any] = {}


class BiencoderModelFactory(ModelFactory):
    """A Biencoder-specific HuggingFace model factory.

    NOTE: This is similar to the HuggingFace HFModelFactory, but it uses `AutoModel`
    instead of `AutoModelForCausalLM` and wraps the result into a `Biencoder`.
    """

    name = "biencoder"
    config: BiencoderModelConfig

    def create_config(self):
        arctic_training_model_config = self.config
        assert isinstance(arctic_training_model_config, BiencoderModelConfig)
        return AutoConfig.from_pretrained(self.config.name_or_path, **arctic_training_model_config.kwargs)

    def create_model(self, model_config: AutoConfig) -> Biencoder:
        arctic_training_model_config = self.config
        assert isinstance(arctic_training_model_config, BiencoderModelConfig)
        trust_remote_code = arctic_training_model_config.kwargs.get("trust_remote_code", None)
        encoder = AutoModel.from_pretrained(
            self.config.name_or_path,
            config=model_config,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=self.config.dtype,
            trust_remote_code=trust_remote_code,
        )
        return Biencoder(encoder, pooling=arctic_training_model_config.pooling)

    def post_create_model_callback(self, model: Biencoder):
        if self.config.peft_config:
            # NOTE: This typecast is technically incorrect but should work in practice.
            peft_config = cast(PeftConfig, self.config.peft_config)
            model.encoder = get_peft_model(model.encoder, peft_config)

        if not self.config.disable_activation_checkpoint:
            model.encoder.gradient_checkpointing_enable()
            model.encoder = HFModelFactory.make_model_gradient_checkpointing_compatible(model.encoder)

        return model
