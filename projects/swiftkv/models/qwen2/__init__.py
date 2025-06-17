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

from .configuration_qwen2_swiftkv import Qwen2SwiftKVConfig
from .modeling_qwen2_swiftkv import Qwen2SwiftKVForCausalLM
from .modeling_qwen2_swiftkv import Qwen2SwiftKVModel


def register_qwen2_swiftkv():
    AutoConfig.register("qwen2_swiftkv", Qwen2SwiftKVConfig)
    AutoModel.register(Qwen2SwiftKVConfig, Qwen2SwiftKVModel)
    AutoModelForCausalLM.register(Qwen2SwiftKVConfig, Qwen2SwiftKVForCausalLM)


__all__ = [
    "Qwen2SwiftKVConfig",
    "Qwen2SwiftKVForCausalLM",
    "Qwen2SwiftKVModel",
    "register_qwen2_swiftkv",
]
