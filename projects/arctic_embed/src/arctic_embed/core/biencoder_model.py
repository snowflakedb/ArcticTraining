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

from __future__ import annotations

import logging
import os
from typing import Callable
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from transformers import PreTrainedModel

PoolingOption = Literal["first_token", "last_token", "mean", "splade"]

logger = logging.getLogger(__name__)


class Biencoder(nn.Module):
    """Model for one-tower text embedding via a transformer `PreTrainedModel`."""

    def __init__(self, encoder: PreTrainedModel, pooling: PoolingOption = "first_token") -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.config = encoder.config
        # Caches for SPLADE pooled weights from the most recent forward.
        self._cached_query_pooled: Optional[Tensor] = None
        self._cached_document_pooled: Optional[Tensor] = None

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # SPLADE-style sparse pooling branches on logits instead of hidden states.
        if self.pooling == "splade":
            if not hasattr(out, "logits"):
                raise ValueError(
                    f"Encoder of class {self.encoder.__class__} must output `logits` for SPLADE pooling."
                )
            logits = out.logits  # (batch, seq_len, vocab_size)
            pooled = splade_pool(logits, attention_mask)
            # NOTE: For SPLADE we avoid L2-normalization to preserve magnitude and sparsity.
            return pooled.contiguous()

        if not hasattr(out, "last_hidden_state"):
            raise ValueError(
                f"Encoder of class {self.encoder.__class__} is missing the "
                "convention of the `forward` function having a `last_hidden_state` "
                "property."
            )
        out = out.last_hidden_state
        assert out.ndim == 3  # batch, token, hidden_dim.
        if self.pooling == "first_token":
            out = first_token_pool(out, attention_mask)
        elif self.pooling == "last_token":
            out = last_token_pool(out, attention_mask)
        elif self.pooling == "mean":
            out = average_pool(out, attention_mask)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        out = F.normalize(out, dim=-1)
        out = out.contiguous()
        return out

    def forward(
        self,
        query_input_ids: Tensor,
        query_attention_mask: Tensor,
        document_input_ids: Tensor,
        document_attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Clear caches from any previous call
        self._cached_query_pooled = None
        self._cached_document_pooled = None

        if self.pooling == "splade":
            # Run encoder to obtain logits for query and cache.
            q_out = self.encoder(input_ids=query_input_ids, attention_mask=query_attention_mask)
            if not hasattr(q_out, "logits"):
                raise ValueError(
                    f"Encoder of class {self.encoder.__class__} must output `logits` for SPLADE pooling."
                )
            self._cached_query_pooled = splade_pool(q_out.logits, query_attention_mask).contiguous()
            query_vectors = self._cached_query_pooled

            # Run encoder to obtain logits for document and cache.
            d_out = self.encoder(input_ids=document_input_ids, attention_mask=document_attention_mask)
            if not hasattr(d_out, "logits"):
                raise ValueError(
                    f"Encoder of class {self.encoder.__class__} must output `logits` for SPLADE pooling."
                )
            self._cached_document_pooled = splade_pool(d_out.logits, document_attention_mask).contiguous()
            document_vectors = self._cached_document_pooled
        else:
            query_vectors = self.encode(query_input_ids, query_attention_mask)
            document_vectors = self.encode(document_input_ids, document_attention_mask)
        return query_vectors, document_vectors

    def compute_splade_flops_doc_cached(self) -> Tensor:
        """SPLADE v2 FLOPs proxy using cached pooled weights (doc side), Eq. (4).

        Let w_j(d_i) be the SPLADE weight of term j for sample i (after activation+pooling).
        With N in-batch samples, Eq. (4): ℓ_FLOPS = Σ_j ( (1/N) Σ_i w_j(d_i) )^2.
        """
        if self.pooling != "splade":
            raise ValueError("FLOPs regularizer requested but pooling is not 'splade'.")
        if self._cached_document_pooled is None:
            raise RuntimeError("Document pooled cache is empty; call forward first.")
        return _splade_flops_batch_mean_squared(self._cached_document_pooled)

    def compute_splade_flops_query_cached(self) -> Tensor:
        """SPLADE v2 FLOPs proxy using cached pooled weights (query side), Eq. (4).

        Same formula as doc side; we apply a different scalar weight in the trainer.
        """
        if self.pooling != "splade":
            raise ValueError("FLOPs regularizer requested but pooling is not 'splade'.")
        if self._cached_query_pooled is None:
            raise RuntimeError("Query pooled cache is empty; call forward first.")
        return _splade_flops_batch_mean_squared(self._cached_query_pooled)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ) -> None:
        self.encoder.save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs,
        )


def average_pool(out: Tensor, attention_mask: Tensor) -> Tensor:
    """Average pool across attended-to tokens (i.e. non-padding tokens)."""
    assert out.ndim == 3
    assert attention_mask.ndim == 2
    out = out.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return out.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def first_token_pool(out: Tensor, attention_mask: Tensor) -> Tensor:
    """Select the first non-padding token representation for each sequence."""
    assert out.ndim == 3
    assert attention_mask.ndim == 2
    batch_size = out.shape[0]
    row = torch.arange(batch_size, device=out.device)
    if attention_mask.dtype == torch.bool:
        attention_mask = attention_mask.to(torch.int8)
    col = attention_mask.argmax(dim=1)  # position of the first non-padding token
    return out[row, col, ...]


def last_token_pool(out: Tensor, attention_mask: Tensor) -> Tensor:
    """Selecting the last non-padding token representation for each sequence."""
    assert out.ndim == 3
    assert attention_mask.ndim == 2
    batch_size = out.shape[0]
    row = torch.arange(batch_size, device=out.device)
    col = attention_mask.sum(dim=1) - 1  # position of the last non-padding token
    return out[row, col, ...]


def splade_pool(logits: Tensor, attention_mask: Tensor) -> Tensor:
    """SPLADE-style sparse pooling over vocabulary logits.

    - Expects `logits` with shape (batch, token, vocab_size)
    - Applies activation: log(1 + relu(logits))
    - Aggregates over tokens with masked max
    """
    assert logits.ndim == 3
    assert attention_mask.ndim == 2
    # Activation as in SPLADE
    activated = torch.log1p(F.relu(logits))
    # Mask out padding positions so they do not affect max
    mask = ~attention_mask[..., None].bool()
    activated = activated.masked_fill(mask, float("-inf"))
    # Max over sequence dimension
    pooled = torch.amax(activated, dim=1)
    # Replace -inf (for fully-masked sequences) with zeros to avoid nans downstream
    pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)
    return pooled

def _splade_flops_batch_mean_squared(pooled_weights: Tensor) -> Tensor:
    """Compute Eq. (4): || (1/N) Σ_i w(d_i) ||_2^2, where w(d_i) are pooled SPLADE weights.

    pooled_weights: (batch, vocab_size)
    returns: scalar Tensor (batch-mean vector squared L2 norm)
    """
    assert pooled_weights.ndim == 2
    mean_vec = pooled_weights.mean(dim=0)
    return (mean_vec.pow(2).sum())
