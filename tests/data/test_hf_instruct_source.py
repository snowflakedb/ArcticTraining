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

import pytest

from arctic_training.data.hf_instruct_source import KNOWN_DATASETS
from .utils import create_data_factory


@pytest.mark.parametrize("dataset_name", list(KNOWN_DATASETS.keys()))
def test_known_datasets_with_hf_instruct_source(model_name: str, dataset_name: str, tmp_path: Path):
    """Test that each known dataset can be loaded with HFDataSourceInstruct via SFTDataFactory."""
    # Use a small subset of data to keep tests fast - split mapping is handled automatically
    dataset_with_subset = f"{dataset_name}:train[:5]"
    
    sft_data_factory = create_data_factory(
        model_name=model_name,
        data_config_kwargs=dict(
            type="sft",
            sources=[{
                "type": "huggingface_instruct", 
                "name_or_path": dataset_with_subset,
            }],
            cache_dir=tmp_path,
            # Use minimal processing to speed up tests
            pack_samples=False,
            filter_samples=False,
        ),
    )
    
    training_dataloader, _ = sft_data_factory()
    assert len(training_dataloader) > 0, f"No data loaded for {dataset_name}"
    
    # Verify the first batch has the expected structure
    first_batch = next(iter(training_dataloader))
    assert "input_ids" in first_batch, f"Missing input_ids in batch for {dataset_name}"
    assert "labels" in first_batch, f"Missing labels in batch for {dataset_name}"