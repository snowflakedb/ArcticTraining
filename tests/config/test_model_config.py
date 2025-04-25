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

import pytest
import torch

from arctic_training.config.model import ModelConfig


@pytest.mark.parametrize(
    "dtype_list",
    [
        (torch.float16, "torch.float16", "fp16", "float16", "half"),
        (torch.float32, "torch.float32", "fp32", "float32", "float"),
        (torch.bfloat16, "torch.bfloat16", "bf16", "bfloat16", "bfloat"),
    ],
)
def test_dtype_field(dtype_list):
    for dtype in dtype_list:
        config = ModelConfig(type="sft", name_or_path="model-name", dtype=dtype)
        assert config.dtype == dtype_list[0]
