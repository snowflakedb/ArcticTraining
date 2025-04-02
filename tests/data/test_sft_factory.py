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

from pathlib import Path
from typing import List

import pytest

from .utils import create_sft_data_factory


@pytest.mark.parametrize(
    "training_sources, expected_sum",
    [
        (["HuggingFaceH4/ultrachat_200k:train[:20]"], 487103798),
        (
            [
                "HuggingFaceH4/ultrachat_200k:train[:20]",
                "Open-Orca/SlimOrca:train[:20]",
            ],
            591408621,
        ),
    ],
)
def test_generated_data(model_name: str, training_sources: List[str], expected_sum: int, tmp_path: Path):
    sft_data_factory = create_sft_data_factory(model_name=model_name, sources=training_sources, cache_dir=tmp_path)
    training_dataloader, _ = sft_data_factory()

    # Quick check that the data is the same as expected. The sum value was
    # generated by running the test initially and then copying the value.
    tensor_sum = 0
    for batch in training_dataloader:
        for key in ("input_ids", "labels", "position_ids"):
            tensor_sum += batch[key].sum().item()

    assert tensor_sum == expected_sum, f"Incorrect tensor sum: {tensor_sum}. Expected {expected_sum}"


def test_sft_factory_cache_path_uniqueness(model_name: str, tmp_path: Path):
    data_sources = [
        "HuggingFaceH4/ultrachat_200k",
        "Open-Orca/SlimOrca",
    ]
    data_factory_1 = create_sft_data_factory(model_name=model_name, sources=data_sources, cache_dir=tmp_path)

    data_sources = data_sources[:1]
    data_factory_2 = create_sft_data_factory(model_name=model_name, sources=data_sources, cache_dir=tmp_path)

    cache_path_1 = data_factory_1.cache_path(data_factory_1._get_data_sources(data_factory_1.config.sources))
    cache_path_2 = data_factory_2.cache_path(data_factory_2._get_data_sources(data_factory_2.config.sources))

    assert cache_path_1 != cache_path_2, "Cache paths were not unique"
