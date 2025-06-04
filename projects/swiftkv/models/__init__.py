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

from .deepseek_v2 import DeepseekV2SwiftKVConfig
from .deepseek_v2 import DeepseekV2SwiftKVForCausalLM
from .deepseek_v2 import DeepseekV2SwiftKVModel
from .deepseek_v2 import register_deepseek_v2_swiftkv
from .llama import LlamaSwiftKVConfig
from .llama import LlamaSwiftKVForCausalLM
from .llama import LlamaSwiftKVModel
from .llama import register_llama_swiftkv
from .qwen2 import Qwen2SwiftKVConfig
from .qwen2 import Qwen2SwiftKVForCausalLM
from .qwen2 import Qwen2SwiftKVModel
from .qwen2 import register_qwen2_swiftkv


def register_all_swiftkv():
    """Register all SwiftKV models."""
    register_deepseek_v2_swiftkv()
    register_llama_swiftkv()
    register_qwen2_swiftkv()


__all__ = [
    "DeepseekV2SwiftKVConfig",
    "DeepseekV2SwiftKVForCausalLM",
    "DeepseekV2SwiftKVModel",
    "LlamaSwiftKVConfig",
    "LlamaSwiftKVForCausalLM",
    "LlamaSwiftKVModel",
    "Qwen2SwiftKVConfig",
    "Qwen2SwiftKVForCausalLM",
    "Qwen2SwiftKVModel",
    "register_all_swiftkv",
    "register_deepseek_v2_swiftkv",
    "register_llama_swiftkv",
    "register_qwen2_swiftkv",
]
