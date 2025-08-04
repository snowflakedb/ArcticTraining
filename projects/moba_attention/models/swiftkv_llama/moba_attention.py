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
MOBA attention integration for SwiftKV LLaMA models.

This module registers MOBA attention as an available attention implementation
that can be used with SwiftKV models by setting _attn_implementation="moba".
"""

from functools import partial
from typing import Optional, Tuple

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

# Import MOBA functionality
try:
    from ...moba.config import MoBAConfig
    from ...moba.moba_efficient import moba_attn_varlen
    from ...moba.wrapper import moba_layer
except ImportError as e:
    raise ImportError(
        f"Failed to import MOBA dependencies. Make sure flash-attn and other dependencies are installed. "
        f"Original error: {e}"
    )


def moba_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    MOBA attention forward function compatible with SwiftKV models.
    
    Args:
        module: The attention module (must have config with moba_chunk_size and moba_topk)
        query: Query tensor [batch, q_heads, q_len, head_dim]
        key: Key tensor [batch, kv_heads, kv_len, head_dim]
        value: Value tensor [batch, kv_heads, kv_len, head_dim]
        attention_mask: Attention mask (unused for MOBA)
        dropout: Dropout probability
        scaling: Attention scaling factor
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (attention_output, None)
    """
    # Get MOBA config from the module's config
    config = getattr(module, 'config', None)
    if config is None:
        raise ValueError("Module must have a config attribute with moba_chunk_size and moba_topk")
    
    moba_config = MoBAConfig(
        moba_chunk_size=getattr(config, 'moba_chunk_size', 4096),
        moba_topk=getattr(config, 'moba_topk', 8)
    )
    
    # Use MOBA layer wrapper
    return moba_layer(
        moba_attn_varlen,
        moba_config,
        module,
        query,
        key,
        value,
        dropout=dropout,
        scaling=scaling,
        **kwargs
    )


def register_moba_attention():
    """Register MOBA as an available attention implementation."""
    if "moba" not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS["moba"] = moba_attention_forward
        print("Registered MOBA attention implementation")


# NOTE: enable_moba_for_model is no longer needed since we have LlamaSwiftKVMoBAForCausalLM
# with MOBA built-in. This function has been removed in favor of the clean inheritance approach. 