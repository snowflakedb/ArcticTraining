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

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_dataset

from .utils import create_data_factory


def test_huggingface_local_data_source(model_name: str, tmp_path: Path):
    # Load small dataset and save locally with "train" split
    dataset_path = tmp_path / "local_ultrachat"
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", streaming=True, split="train_sft")
    dataset = Dataset.from_list(list(dataset.take(20)), features=dataset.features)
    DatasetDict(dict(train=dataset)).save_to_disk(dataset_path.as_posix())

    # Load saved dataset using HuggingFaceLocal data source
    sft_data_factory = create_data_factory(
        model_name=model_name,
        data_config_kwargs=dict(
            type="sft",
            sources=[dataset_path.as_posix()],
            cache_dir=tmp_path,
        ),
    )
    training_dataloader, _ = sft_data_factory()

    assert len(training_dataloader) > 0, "No data loaded"


def test_huggingface_local_data_source_no_split(model_name: str, tmp_path: Path):
    # Load small dataset and save locally without splits
    dataset_path = tmp_path / "local_ultrachat"
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", streaming=True, split="train_sft")
    dataset = Dataset.from_list(list(dataset.take(20)), features=dataset.features)
    dataset.save_to_disk(dataset_path.as_posix())

    # Load saved dataset using HuggingFaceLocal data source
    sft_data_factory = create_data_factory(
        model_name=model_name,
        data_config_kwargs=dict(
            type="sft",
            sources=[dataset_path.as_posix()],
            cache_dir=tmp_path,
        ),
    )
    training_dataloader, _ = sft_data_factory()

    assert len(training_dataloader) > 0, "No data loaded"
