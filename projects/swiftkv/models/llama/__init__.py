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

from .configuration_llama_swiftkv import LlamaSwiftKVConfig
from .modeling_llama_swiftkv import LlamaSwiftKVForCausalLM
from .modeling_llama_swiftkv import LlamaSwiftKVModel


def register_llama_swiftkv():
    AutoConfig.register("llama_swiftkv", LlamaSwiftKVConfig)
    AutoModel.register(LlamaSwiftKVConfig, LlamaSwiftKVModel)
    AutoModelForCausalLM.register(LlamaSwiftKVConfig, LlamaSwiftKVForCausalLM)


__all__ = [
    "LlamaSwiftKVConfig",
    "LlamaSwiftKVForCausalLM",
    "LlamaSwiftKVModel",
    "register_llama_swiftkv",
]
