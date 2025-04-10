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

PoolingOption = Literal["first_token", "last_token", "mean"]

logger = logging.getLogger(__name__)


class Biencoder(nn.Module):
    """Model for one-tower text embedding via a transformer `PreTrainedModel`."""

    def __init__(self, encoder: PreTrainedModel, pooling: PoolingOption = "first_token") -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
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
        query_vectors = self.encode(query_input_ids, query_attention_mask)
        document_vectors = self.encode(document_input_ids, document_attention_mask)
        return query_vectors, document_vectors

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
