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
SwiftKV + MOBA model implementation.

This module provides LlamaSwiftKVMoBAForCausalLM that inherits from 
LlamaSwiftKVForCausalLM and has MOBA attention built-in.
"""

from typing import Optional, Union

from ....swiftkv.models.llama.modeling_llama_swiftkv import LlamaSwiftKVForCausalLM, LlamaSwiftKVModel
from .configuration_swiftkv_moba import LlamaSwiftKVMoBAConfig
from .moba_attention import register_moba_attention


class LlamaSwiftKVMoBAModel(LlamaSwiftKVModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`LlamaSwiftKVDecoderLayer`].

    Args:
        config: LlamaSwiftKVConfig
    """

    config_class = LlamaSwiftKVMoBAConfig

class LlamaSwiftKVMoBAForCausalLM(LlamaSwiftKVForCausalLM):
    """
    LLaMA model with SwiftKV and MOBA optimizations for Causal Language Modeling.
    
    Inherits from LlamaSwiftKVForCausalLM and automatically enables MOBA attention.
    """
    
    config_class = LlamaSwiftKVMoBAConfig
    
    def __init__(self, config):
        # Validate config type
        if not isinstance(config, LlamaSwiftKVMoBAConfig):
            raise TypeError(f"Expected LlamaSwiftKVMoBAConfig, got {type(config)}")
        
        # Register MOBA attention before initializing
        register_moba_attention()
        
        # Set MOBA as the attention implementation (avoid mutating the original config)
        if not hasattr(config, '_attn_implementation') or config._attn_implementation != "moba":
            config._attn_implementation = "moba"
        
        # Initialize parent class (SwiftKV model)
        super().__init__(config)
        
        print(f"Initialized SwiftKV + MOBA model with chunk_size={config.moba_chunk_size}, topk={config.moba_topk}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load a pretrained model and convert it to SwiftKV + MOBA.
        
        Args:
            pretrained_model_name_or_path: Model to load
            *args, **kwargs: Arguments passed to parent from_pretrained
        """
        # Ensure we have the right config class
        config = kwargs.get('config', None)
        if config is None or not isinstance(config, LlamaSwiftKVMoBAConfig):
            # Create SwiftKV + MOBA config from the base model
            from transformers import AutoConfig
            base_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            
            # Extract MOBA parameters from kwargs if provided
            moba_chunk_size = kwargs.pop('moba_chunk_size', 4096)
            moba_topk = kwargs.pop('moba_topk', 8)
            
            # Extract SwiftKV parameters from kwargs if provided
            swiftkv = kwargs.pop('swiftkv', True)
            num_key_value_layers = kwargs.pop('num_key_value_layers', None)
            key_value_group_size = kwargs.pop('key_value_group_size', 1)
            
            # Validate parameter types
            if not isinstance(moba_chunk_size, int) or moba_chunk_size <= 0:
                raise ValueError(f"moba_chunk_size must be a positive integer, got {moba_chunk_size}")
            if not isinstance(moba_topk, int) or moba_topk <= 0:
                raise ValueError(f"moba_topk must be a positive integer, got {moba_topk}")
            
            config = LlamaSwiftKVMoBAConfig(
                **base_config.to_dict(),
                swiftkv=swiftkv,
                num_key_value_layers=num_key_value_layers,
                key_value_group_size=key_value_group_size,
                moba_chunk_size=moba_chunk_size,
                moba_topk=moba_topk,
            )
            kwargs['config'] = config
        
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
    
    def enable_swiftkv(self, enable: bool = True):
        """Enable or disable SwiftKV optimization."""
        self.config.swiftkv = enable
        return self
    
    def update_moba_config(self, chunk_size: Optional[int] = None, topk: Optional[int] = None):
        """Update MOBA configuration."""
        if chunk_size is not None:
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")
            self.config.moba_chunk_size = chunk_size
        if topk is not None:
            if not isinstance(topk, int) or topk <= 0:
                raise ValueError(f"topk must be a positive integer, got {topk}")
            self.config.moba_topk = topk
        
        print(f"Updated MOBA config: chunk_size={self.config.moba_chunk_size}, topk={self.config.moba_topk}")
        return self 