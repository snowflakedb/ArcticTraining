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

# This file was original taken from the following:
# https://github.com/MoonshotAI/MoBA/tree/61e456bc956c5a25fd9c84e5496b661329cb1b72
# Modification may have been made by Snowflake

from typing import Callable
from typing import Optional
from typing import Tuple

import torch
from flash_attn import flash_attn_func

from .config import MoBAConfig


def hf_to_fa(x: torch.Tensor):
    """
    Args:
        x (torch.Tensor): [batch, heads, seqlen, head_dim]

    Returns:
        torch.Tensor: [batch * seqlen, heads, head_dim]
    """
    return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])


def fa_to_hf(x: torch.Tensor, batch: int):
    """
    Args:
        x (torch.Tensor): [batch * seqlen, heads, head_dim]

    Returns:
        torch.Tensor: [batch, heads, seqlen, head_dim]
    """
    return x.view(batch, -1, x.shape[1], x.shape[2]).permute(0, 2, 1, 3)


def moba_layer(
    moba_impl: Callable,
    moba_config: MoBAConfig,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    Args:
        query (torch.Tensor): [batch, q_heads, q_len, head_dim]
        key (torch.Tensor): [batch, kv_heads, kv_len, head_dim]
        value (torch.Tensor): [batch, kv_heads, kv_len, head_dim]

    Returns:
        attn_output (torch.Tensor): [batch, q_len, q_heads, head_dim]
        attn_weights (None): not needed
    """
    assert module.is_causal
    batch, q_heads, q_len, head_dim = query.shape
    _, kv_heads, kv_len, _ = key.shape
    if q_len == kv_len:
        # prefill phase
        query = hf_to_fa(query)
        key = hf_to_fa(key)
        value = hf_to_fa(value)
        kv_replicas = q_heads // kv_heads
        key = torch.repeat_interleave(key, kv_replicas, dim=1)
        value = torch.repeat_interleave(value, kv_replicas, dim=1)
        cu_seqlens_k = torch.cumsum(
            torch.tensor([0] + [kv_len] * batch, device=query.device),
            dim=0,
            dtype=torch.int32,
        )
        out = moba_impl(
            q=query,
            k=key,
            v=value,
            cu_seqlens=cu_seqlens_k,
            max_seqlen=kv_len,
            moba_chunk_size=moba_config.moba_chunk_size,
            moba_topk=moba_config.moba_topk,
        )
    else:
        # decode phase
        # TODO release paged attn implementation
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        out = flash_attn_func(query, key, value, dropout, scaling, True)
    # out = fa_to_hf(out, batch)
    return out, None
