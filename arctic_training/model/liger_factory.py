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

import importlib.metadata

from packaging import version
from transformers import PreTrainedModel

from arctic_training.model.hf_factory import HFModelFactory


class LigerModelFactory(HFModelFactory):
    name = "liger"

    def create_model(self, model_config) -> PreTrainedModel:
        try:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
        except ImportError:
            raise ImportError(
                "You need to install the liger-kernel package to use LigerKernel models: `pip install liger-kernel`"
            )
        liger_version_min = "0.5.5"  # earlier versions were silently dropping the attn_implementation kwargs
        liger_version_have = importlib.metadata.version("liger_kernel")
        if version.parse(liger_version_have) < version.parse(liger_version_min):
            raise ValueError(
                f"liger-kernel>={liger_version_min} is required, but you have liger-kernel=={liger_version_have}"
            )

        return AutoLigerKernelForCausalLM.from_pretrained(
            self.config.name_or_path,
            config=model_config,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=self.config.dtype.value,
        )
