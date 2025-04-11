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

import math
from functools import partial

import flash_attn
import torch

from .moba_efficient import moba_attn_varlen

original_flash_attn_varlen_func = flash_attn.flash_attn_varlen_func


def moba_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    moba=False,
    moba_chunk_size=2048,
    moba_topk=16,
):
    if moba:
        assert torch.equal(
            cu_seqlens_q, cu_seqlens_k
        ), f"{cu_seqlens_q} and {cu_seqlens_k} MoBA Only supports Self Attention"
        assert max_seqlen_q == max_seqlen_k, "MoBA Only supports Self Attention"
        assert (
            dropout_p == 0.0
            and (softmax_scale is None or math.isclose(softmax_scale, 1.0 / q.shape[-1] ** 0.5))
            and causal is True
            and window_size == (-1, -1)
            and softcap == 0.0
            and alibi_slopes is None
            and deterministic is False
            and return_attn_probs is False
            and block_table is None
        ), "Only these paramreters are supported by MoBA"

        cu_seqlens = cu_seqlens_q
        max_seqlen = max_seqlen_q
        return moba_attn_varlen(q, k, v, cu_seqlens, max_seqlen, moba_chunk_size, moba_topk)
    else:
        global original_flash_attn_varlen_func
        return original_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,  # -1 means infinite context window
            softcap=softcap,  # 0.0 means deactivated
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            block_table=block_table,
        )


def patch_flash_attn_varlen_func_for_moba(chunk_size, topk):
    import transformers.modeling_flash_attention_utils

    transformers.modeling_flash_attention_utils.flash_attn_varlen_func = partial(
        moba_flash_attn_varlen_func,
        moba=True,
        moba_chunk_size=chunk_size,
        moba_topk=topk,
    )
