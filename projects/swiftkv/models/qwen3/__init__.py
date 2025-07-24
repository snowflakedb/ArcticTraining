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

from .configuration_qwen3_swiftkv import Qwen3SwiftKVConfig
from .modeling_qwen3_swiftkv import Qwen3SwiftKVForCausalLM
from .modeling_qwen3_swiftkv import Qwen3SwiftKVModel


def register_qwen3_swiftkv():
    AutoConfig.register("qwen3_swiftkv", Qwen3SwiftKVConfig)
    AutoModel.register(Qwen3SwiftKVConfig, Qwen3SwiftKVModel)
    AutoModelForCausalLM.register(Qwen3SwiftKVConfig, Qwen3SwiftKVForCausalLM)


__all__ = [
    "Qwen3SwiftKVConfig",
    "Qwen3SwiftKVForCausalLM",
    "Qwen3SwiftKVModel",
    "register_qwen3_swiftkv",
]
