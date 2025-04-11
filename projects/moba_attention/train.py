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

from projects.moba_attention.moba import patch_flash_attn_varlen_func_for_moba
from projects.swiftkv.train import SwiftKVModelConfig
from projects.swiftkv.train import SwiftKVModelFactory
from projects.swiftkv.train import SwiftKVTrainer
from projects.swiftkv.train import SwiftKVTrainerConfig


class MoBAModelConfig(SwiftKVModelConfig):
    num_key_value_layers: int
    key_value_group_size: int = 1
    moba_chunk_size: int = 4096
    moba_topk: int = 8
    attn_implementation: str = "flash_attention_2"


class MoBAModelFactory(SwiftKVModelFactory):
    name = "moba"
    config: MoBAModelConfig

    def post_create_config_callback(self, hf_config):

        hf_config = super().post_create_config_callback(hf_config)
        hf_config.moba_chunk_size = self.config.moba_chunk_size
        hf_config.moba_topk = self.config.moba_topk

        # The code path is so convoluted I do not know which one to set
        # So setting both of them.
        hf_config._attn_implementation = self.config.attn_implementation
        hf_config.attn_implementation = self.config.attn_implementation

        """patch the flash attention in the model to use moba
        there is an assumption that this method will be called
        before model creation"""
        patch_flash_attn_varlen_func_for_moba(self.config.moba_chunk_size, self.config.moba_topk)

        return hf_config


class MoBATrainerConfig(SwiftKVTrainerConfig):
    pass


class MoBATrainer(SwiftKVTrainer):
    name = "moba"
    config: MoBATrainerConfig
    model_factory: MoBAModelFactory
