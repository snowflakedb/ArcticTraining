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

import pytest
import torch
from arctic_embed.core.slice_sparse_coo_tensor import slice_sparse_coo_tensor

test_tensor = torch.tensor([[0, 1, 0, 2], [3, 0, 0, 5], [0, 0, 4, 6], [1, 1, 0, 0], [0, 0, 0, 0]])
test_tensor_sparse = test_tensor.to_sparse_coo()
test_slices = [
    [(0, 2), (0, 2)],
    [(2, 4), (0, 2)],
    [(0, 4), (None, 2)],
    [(1, None), (1, None)],
]


@pytest.mark.parametrize("slice_list", test_slices)
def test_slice_sparse_coo_tensor(
    slice_list: Sequence[Tuple[Optional[int], Optional[int]]],
) -> None:
    sliced_tensor_sparse = slice_sparse_coo_tensor(test_tensor_sparse, slice_list)
    sliced_tensor = test_tensor[[slice(start, end) for start, end in slice_list]]
    assert torch.equal(sliced_tensor_sparse.to_dense(), sliced_tensor)
