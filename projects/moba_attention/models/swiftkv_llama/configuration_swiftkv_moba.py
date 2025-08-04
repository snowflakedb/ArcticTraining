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

from typing import Optional

from ....swiftkv.models.llama.configuration_llama_swiftkv import LlamaSwiftKVConfig


class LlamaSwiftKVMoBAConfig(LlamaSwiftKVConfig):
    """
    Configuration class for LLaMA SwiftKV with MOBA attention.
    
    Inherits from LlamaSwiftKVConfig and adds MOBA-specific parameters.
    
    Args:
        moba_chunk_size (int, optional):
            The chunk size for MOBA attention. Default is 4096.
        moba_topk (int, optional):
            The top-k value for MOBA attention. Default is 8.
    """

    model_type = "llama_swiftkv_moba"

    def __init__(
        self,
        moba_chunk_size: int = 4096,
        moba_topk: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # MOBA parameters
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk 