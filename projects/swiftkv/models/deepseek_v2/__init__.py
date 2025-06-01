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

from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForCausalLM

from .configuration_deepseek import DeepseekV2Config
from .configuration_deepseek_swiftkv import DeepseekV2SwiftKVConfig
from .modeling_deepseek import DeepseekV2ForCausalLM
from .modeling_deepseek import DeepseekV2Model
from .modeling_deepseek_swiftkv import DeepseekV2SwiftKVForCausalLM
from .modeling_deepseek_swiftkv import DeepseekV2SwiftKVModel


def register_deepseek_v2():
    # Register original deepseek_v2 model and config since it's not in transformers
    AutoConfig.register("deepseek_v2", DeepseekV2Config)
    AutoModel.register(DeepseekV2Config, DeepseekV2Model)
    AutoModelForCausalLM.register(DeepseekV2Config, DeepseekV2ForCausalLM)


def register_deepseek_v2_swiftkv():
    # Register the swiftkv version of deepseek_v2
    AutoConfig.register("deepseek_v2_swiftkv", DeepseekV2SwiftKVConfig)
    AutoModel.register(DeepseekV2SwiftKVConfig, DeepseekV2SwiftKVModel)
    AutoModelForCausalLM.register(DeepseekV2SwiftKVConfig, DeepseekV2SwiftKVForCausalLM)


__all__ = [
    "DeepseekV2SwiftKVConfig",
    "DeepseekV2SwiftKVForCausalLM",
    "DeepseekV2SwiftKVModel",
    "register_deepseek_v2",
    "register_deepseek_v2_swiftkv",
]
