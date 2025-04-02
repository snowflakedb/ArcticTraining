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

"""Losses for training biencoders."""

from math import ceil
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor


class TruncatedDimensionLoss(NamedTuple):
    dim: int
    loss: Tensor


def one_size_truncated_mrl_info_nce_loss(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    relations: Tensor,
    truncated_dimension: Optional[int] = None,
    mrl_weight: float = 0.5,
    temperature: float = 0.01,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """InfoNCE loss with one truncated dimension of Matroshka Representation Learning
    (MRL) loss.

    NOTE: Uses activation checkpointing to reduce memory usage that can happen at
    very large batch size (e.g. 32k batch size takes ~4GB of memory just for the
    score matrix).
    """
    assert 0 < mrl_weight < 1
    full_dim_weight = 1 - mrl_weight
    full_dimension = query_embeddings.size(1)
    assert truncated_dimension is None or 0 < truncated_dimension < full_dimension
    loss_full_dim = torch.utils.checkpoint.checkpoint(
        _dim_truncated_infonce,
        use_reentrant=False,
        query_embeddings=query_embeddings,
        document_embeddings=document_embeddings,
        relations=relations,
        dim=full_dimension,
        temperature=temperature,
    )
    if truncated_dimension is None:
        loss_truncated_dim = None
        loss = loss_full_dim
    else:
        loss_truncated_dim = torch.utils.checkpoint.checkpoint(
            _dim_truncated_infonce,
            use_reentrant=False,
            query_embeddings=query_embeddings,
            document_embeddings=document_embeddings,
            relations=relations,
            dim=truncated_dimension,
            temperature=temperature,
        )
        loss = full_dim_weight * loss_full_dim + mrl_weight * loss_truncated_dim
    return loss, loss_full_dim, loss_truncated_dim


def _dim_truncated_infonce(
    query_embeddings: Tensor,
    document_embeddings: Tensor,
    relations: Tensor,
    dim: int,
    temperature: float,
) -> Tensor:
    q_emb = F.normalize(query_embeddings[:, :dim], dim=1)
    d_emb = F.normalize(document_embeddings[:, :dim], dim=1)
    scores = torch.matmul(q_emb, d_emb.transpose(0, 1))
    loss = info_nce_loss(scores, relations=relations, temperature=temperature)
    return loss


def info_nce_loss(scores: Tensor, relations: Tensor, temperature: float = 0.01) -> Tensor:
    """InfoNCE loss for potentially many-to-many query-document pairings.

    - `scores` is a floating point matrix of shape (n_query, n_doc)
    - `relations` is an integer matrix of shape (n_query, n_doc)
        - `relations[i, j] == 1` --> query i and document j are a positive match
        - `relations[i, j] == -1` --> query i and document j are a negative match
        - `relations[i, j] == 0` --> no labeled relation between query i and document j

    """
    result = _MemoryEfficientInfoNCE.apply(scores, relations, temperature)  # type: ignore  # noqa: E501
    return result


class _MemoryEfficientInfoNCE(torch.autograd.Function):
    """A memory-efficient InfoNCE implementation which allows reindexing to avoid
    embedding duplicate queries/documents shared across query-doc pairs but which
    also avoids doing that indexing / `torch.nonzero` inside the computational
    graph and sub-batches the forward pass to lower memory impact.
    """

    FWD_SUB_BATCH_SIZE = 1024

    @staticmethod
    def forward(ctx, scores: Tensor, relations: Tensor, temperature: float) -> Tensor:  # type: ignore
        # If there are most then one positive column items per row item, we must
        # expand those rows into multiple rows, each with a single positive,
        # marking the other positive items as un-labeled (i.e. ignored).
        pos_row_idx, pos_col_idx = _torch_safe_where(relations == 1)
        is_expanded = pos_row_idx.size(0) != scores.size(0)
        if is_expanded:
            # Work on a copy of the original matrix.
            relations_expanded = relations.clone()
            # Zero out positives in-place.
            relations_expanded[pos_row_idx, pos_col_idx] = 0
            # Expand to one row per positives.
            relations_expanded = relations_expanded[pos_row_idx]
            # Un-zero-out one positive per row.
            exploded_pos_row_idx = torch.arange(pos_row_idx.size(0))
            relations_expanded[exploded_pos_row_idx, pos_col_idx] = 1
            # Expand the scores to one row per positives.
            scores_expanded = scores[pos_row_idx]
            # Create a matrix to un-explode the gradient by combining exploded
            # rows' gradient entries.
            unexpansion_matrix = torch.zeros(
                (scores.size(0), pos_row_idx.size(0)),
                dtype=scores.dtype,
                device=scores.device,
            )
            unexpansion_matrix[pos_row_idx, exploded_pos_row_idx] = 1.0
        else:
            scores_expanded = scores
            relations_expanded = relations
            unexpansion_matrix = torch.tensor(torch.nan)

        # Calculate the loss.
        logits = scores_expanded.masked_fill(relations_expanded == 0, -1e12) / temperature
        # Do batched cross entropy to avoid memory pressure.
        # result = F.cross_entropy(input=logits, target=pos_col_idx)
        result = torch.tensor(0.0, device=scores.device)
        sub_batch_size = _MemoryEfficientInfoNCE.FWD_SUB_BATCH_SIZE
        for start in range(0, logits.size(0), sub_batch_size):
            end = start + sub_batch_size
            result += F.cross_entropy(input=logits[start:end], target=pos_col_idx[start:end], reduction="sum")
        result /= logits.size(0)
        ctx.save_for_backward(pos_col_idx, logits, unexpansion_matrix)
        ctx.temperature = temperature
        ctx.is_expanded = is_expanded
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):  # type: ignore
        pos_col_idx, logits, unexpansion_matrix = ctx.saved_tensors
        # Reference on gradient of cross-entropy loss on softmax:
        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        d_grad_d_s_expanded = grad_output * (
            (F.softmax(logits, dim=-1) - F.one_hot(pos_col_idx, num_classes=logits.size(1)))
            / logits.size(0)
            / ctx.temperature
        )
        if ctx.is_expanded:
            d_grad_ds_s = unexpansion_matrix @ d_grad_d_s_expanded
        else:
            d_grad_ds_s = d_grad_d_s_expanded
        return d_grad_ds_s, None, None


def _torch_safe_where(t: torch.Tensor) -> Sequence[torch.Tensor]:
    """A safe version of `torch.where` that works with large tensors having
    over INT_MAX elements by executing chunk-wise over the first dimension.

    See bug: https://github.com/pytorch/pytorch/issues/51871
    Based on solution: https://github.com/facebookresearch/segment-anything/pull/569
    NOTE: Bugfixed in November 2024, but older PyTorch/CUDA versions still need this
        workaround.
    """
    int_max = torch.iinfo(torch.int32).max
    num_elements = t.numel()
    if num_elements < int_max:
        return torch.where(t)
    else:
        num_chunks = ceil(num_elements / int_max)
        chunk_size = ceil(t.size(0) / num_chunks)
        result_chunks: Tuple[List[torch.Tensor], ...] = tuple([] for _ in range(t.ndim))
        for start in range(0, t.size(0), chunk_size):
            chunk = t[start : start + chunk_size, :]
            chunk_res = list(torch.where(chunk))
            chunk_res[0] += start
            for axis in range(t.ndim):
                result_chunks[axis].append(chunk_res[axis])

        return tuple(torch.cat(axis_res, dim=0) for axis_res in result_chunks)
