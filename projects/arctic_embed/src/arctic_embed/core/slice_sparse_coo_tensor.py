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
from typing import Sequence
from typing import Tuple

import torch


def slice_sparse_coo_tensor(
    sparse_tensor: torch.Tensor,
    start_end_pairs: Sequence[Tuple[Optional[int], Optional[int]]],
) -> torch.Tensor:
    assert len(start_end_pairs) <= sparse_tensor.ndim
    sparse_tensor = sparse_tensor.coalesce()
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    mask = torch.ones_like(values, dtype=torch.bool)
    start_by_dim = torch.zeros((len(start_end_pairs), 1), dtype=torch.int64)
    size = []
    for i, (start, end) in enumerate(start_end_pairs):
        if start is not None:
            start_by_dim[i, 0] = start
            mask = mask & (indices[i] >= start)
        if end is not None:
            mask = mask & (indices[i] < end)
        dim_size = (end if end is not None else sparse_tensor.size(i)) - (start if start is not None else 0)
        size.append(dim_size)
    sliced_indices = indices[:, mask] - start_by_dim
    sliced_values = values[mask]
    return torch.sparse_coo_tensor(indices=sliced_indices, values=sliced_values, size=size)
