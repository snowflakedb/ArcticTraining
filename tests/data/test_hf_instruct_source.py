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

from arctic_training.data.hf_instruct_source import KNOWN_HF_INSTRUCT_DATASETS

from .utils import create_data_factory


def test_known_datasets(model_name: str, tmp_path: Path):
    pytest.skip("Skipping due to limited disk space on Github runners")
    skip_datasets = ["lmsys/lmsys-chat-1m"]  # gated dataset
    data_sources = [f"{dataset}:train[:20]" for dataset in KNOWN_HF_INSTRUCT_DATASETS if dataset not in skip_datasets]
    sft_data_factory = create_data_factory(
        model_name=model_name, data_config_kwargs=dict(type="sft", sources=data_sources, cache_dir=tmp_path)
    )
    training_dataloader, _ = sft_data_factory()
    assert len(training_dataloader) > 0
