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
        liger_version_min = "0.6.1"  # int64 indexing
        liger_version_have = importlib.metadata.version("liger_kernel")
        if version.parse(liger_version_have) < version.parse(liger_version_min):
            raise ValueError(
                f"liger-kernel>={liger_version_min} is required, but you have liger-kernel=={liger_version_have}"
            )

        # Disable liger's mlp override if we are using our mlp override
        swiglu = False if self.trainer.config.tiled_mlp_compute else True
        # XXX: it might be possible to combine the 2 in the future to benefit from the efficient liger swiglu kernel, but currently liger monkey patches the MLP class and thus we would have a race condition on who gets the override.

        try:
            return AutoLigerKernelForCausalLM.from_pretrained(
                self.config.name_or_path,
                config=model_config,
                attn_implementation=self.config.attn_implementation,
                dtype=self.config.dtype.value,
                swiglu=swiglu,
            )
        except KeyError as e:
            raise ValueError(
                f"It appears that liger-kernel=={liger_version_have} doesn't support the architecture of"
                f" {self.config.name_or_path}, perhaps try the latest liger-kernel version. It can't find {e}."
            )
        except Exception:
            raise
