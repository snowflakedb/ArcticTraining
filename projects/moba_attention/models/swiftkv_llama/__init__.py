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

"""
SwiftKV + MOBA integration for LLaMA models.

This module provides a clean way to combine SwiftKV's efficient KV caching
with MOBA's attention optimization for long sequences.
"""

from typing import Optional

from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForCausalLM

# Import SwiftKV base classes
from ....swiftkv.models.llama import (
    register_llama_swiftkv,
    LlamaSwiftKVConfig,
    LlamaSwiftKVForCausalLM,
    LlamaSwiftKVModel,
)

# Import MOBA functionality
from .configuration_swiftkv_moba import LlamaSwiftKVMoBAConfig
from .modeling_swiftkv_moba import LlamaSwiftKVMoBAForCausalLM, LlamaSwiftKVMoBAModel
from .moba_attention import register_moba_attention


def register_llama_swiftkv_moba():
    """Register SwiftKV + MOBA models with transformers."""
    # First register base SwiftKV models
    register_llama_swiftkv()
    
    # Then register SwiftKV + MOBA models
    AutoConfig.register("llama_swiftkv_moba", LlamaSwiftKVMoBAConfig)
    AutoModel.register(LlamaSwiftKVMoBAConfig, LlamaSwiftKVMoBAModel)
    AutoModelForCausalLM.register(LlamaSwiftKVMoBAConfig, LlamaSwiftKVMoBAForCausalLM)
    
    # Register MOBA attention
    register_moba_attention()


def create_swiftkv_moba_model(
    base_model_name_or_path: str,
    swiftkv: bool = True,
    num_key_value_layers: Optional[int] = None,
    key_value_group_size: int = 1,
    moba_chunk_size: int = 4096,
    moba_topk: int = 8,
    **model_kwargs
):
    """
    Create a SwiftKV model with MOBA attention enabled.
    
    Args:
        base_model_name_or_path: Base model to load
        swiftkv: Enable SwiftKV optimization
        num_key_value_layers: Number of layers with full KV computation
        key_value_group_size: Size of KV sharing groups
        moba_chunk_size: MOBA chunk size
        moba_topk: MOBA top-k value
        **model_kwargs: Additional arguments for model loading
        
    Returns:
        Model with SwiftKV and MOBA enabled
    """
    # Validate parameters
    if not isinstance(moba_chunk_size, int) or moba_chunk_size <= 0:
        raise ValueError(f"moba_chunk_size must be a positive integer, got {moba_chunk_size}")
    if not isinstance(moba_topk, int) or moba_topk <= 0:
        raise ValueError(f"moba_topk must be a positive integer, got {moba_topk}")
    
    # Register our models
    register_llama_swiftkv_moba()
    
    # Create config with SwiftKV and MOBA parameters
    config = LlamaSwiftKVMoBAConfig.from_pretrained(
        base_model_name_or_path,
        swiftkv=swiftkv,
        num_key_value_layers=num_key_value_layers,
        key_value_group_size=key_value_group_size,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
    )
    
    # Create model using our specialized class
    model = LlamaSwiftKVMoBAForCausalLM.from_pretrained(
        base_model_name_or_path,
        config=config,
        **model_kwargs
    )
    
    return model


def create_swiftkv_moba_config(
    base_model_name_or_path: str,
    swiftkv: bool = True,
    num_key_value_layers: Optional[int] = None,
    key_value_group_size: int = 1,
    moba_chunk_size: int = 4096,
    moba_topk: int = 8,
    **config_kwargs
):
    """Create a SwiftKV + MOBA configuration."""
    return LlamaSwiftKVMoBAConfig.from_pretrained(
        base_model_name_or_path,
        swiftkv=swiftkv,
        num_key_value_layers=num_key_value_layers,
        key_value_group_size=key_value_group_size,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
        **config_kwargs
    )


__all__ = [
    # Configuration and Model classes
    "LlamaSwiftKVMoBAConfig",
    "LlamaSwiftKVMoBAForCausalLM",
    
    # Model creation functions
    "create_swiftkv_moba_model",
    "create_swiftkv_moba_config",
    
    # Registration functions
    "register_llama_swiftkv_moba",
    "register_moba_attention",
    
    # Re-exported from SwiftKV
    "LlamaSwiftKVForCausalLM",
    "LlamaSwiftKVModel",
    "LlamaSwiftKVConfig",
] 