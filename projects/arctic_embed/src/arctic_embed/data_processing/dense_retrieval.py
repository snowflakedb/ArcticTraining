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

"""Pytorch implementation of GPU-accelerated batch dense retrieval."""

import numpy as np
import torch
from numpy.typing import NDArray
from tqdm.auto import tqdm

from .utils import NDArrayOfFloat


@torch.inference_mode()
def dense_retrieval(
    query_embeddings: NDArrayOfFloat,
    item_embeddings: NDArrayOfFloat,
    retrieval_depth: int = 300,
    item_slice_size: int = 1024000,
    query_slice_size: int = 4096,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Perform dense retrieval with a set of ids and embeddings to get query results."""
    num_queries, num_items = query_embeddings.shape[0], item_embeddings.shape[0]
    retrieval_depth = min(retrieval_depth, num_items)
    result_indices = np.empty((num_queries, retrieval_depth), dtype=np.int32)
    result_scores = np.empty((num_queries, retrieval_depth), dtype=np.float32)

    with tqdm(total=num_queries, desc="dense retrieval", unit="query") as pbar:
        # Go one chunk of queries at a time to keep memory pressure manageable.
        for i, query_start in enumerate(range(0, num_queries, query_slice_size)):
            # Take a slice of queries.
            query_end = query_start + query_slice_size
            query_slice = query_embeddings[query_start:query_end]
            q_tensor = torch.tensor(query_slice)
            if torch.cuda.is_available():
                q_tensor = q_tensor.cuda()

            # Perform retrieval slice-by-slice on the documents using this query slice.
            total_top = retrieval_depth * int(np.ceil(num_items / item_slice_size))
            top_indices = np.empty((query_slice.shape[0], total_top), dtype=np.int32)
            top_scores = np.empty((query_slice.shape[0], total_top), dtype=np.float32)
            for j, item_start in enumerate(range(0, num_items, item_slice_size)):
                item_end = item_start + item_slice_size
                item_slice = item_embeddings[item_start:item_end]
                i_tensor = torch.tensor(item_slice)
                if torch.cuda.is_available():
                    i_tensor = i_tensor.cuda()

                # For the current slices, compute dot product and find top-k indices.
                scores_slice = q_tensor @ i_tensor.T
                topk = torch.topk(scores_slice, k=retrieval_depth)

                # Store these values in the correct location in the arrays that track
                # scores across all of the item slices.
                result_start = j * retrieval_depth
                result_end = result_start + retrieval_depth
                top_indices[:, result_start:result_end] = item_start + topk.indices.cpu().numpy()
                top_scores[:, result_start:result_end] = topk.values.cpu().numpy()

            # Reduce to the global top-k indices and scores across all item slices.
            idx_sort = np.argsort(-top_scores, axis=1)
            top_indices = np.take_along_axis(top_indices, idx_sort, axis=1)[:, :retrieval_depth]
            top_scores = np.take_along_axis(top_scores, idx_sort, axis=1)[:, :retrieval_depth]

            # Store the results.
            result_start = i * query_slice_size
            result_end = result_start + query_slice_size
            result_indices[result_start:result_end] = top_indices
            result_scores[result_start:result_end] = top_scores

            pbar.update(query_slice.shape[0])

    return result_indices, result_scores
