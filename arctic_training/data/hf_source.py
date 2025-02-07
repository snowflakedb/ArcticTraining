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

from typing import Any
from typing import Dict

from datasets import Dataset
from datasets import load_dataset

from arctic_training.config.data import DataSourceConfig
from arctic_training.data.source import DataSource
from arctic_training.registry import register


class HFDataSourceConfig(DataSourceConfig):
    dataset_name: str = ""
    """ Name of the dataset to load. """

    kwargs: Dict[str, Any] = {}
    """ Keyword arguments to pass to the datasets.load_dataset function. """


def set_dataset_name(self, config: HFDataSourceConfig) -> HFDataSourceConfig:
    """Set the dataset name from the config. This is a helper function for the dataset-specific classes we define below."""
    if not config.dataset_name:
        if self.name == "huggingface":
            raise ValueError(
                "Must provide a dataset name for HuggingFace data sources."
            )
        config.dataset_name = self.name
    return config


@register
class HFDataSource(DataSource):
    """Base DataSource class for loading data with HuggingFace datasets library."""

    name = "huggingface"
    config_type = HFDataSourceConfig
    callbacks = [("pre-init", set_dataset_name)]

    def load(self, config: HFDataSourceConfig, split: str) -> Dataset:
        return load_dataset(config.dataset_name, split=split, **config.kwargs)


@register
class UltraChat200K(HFDataSource):
    name = "HuggingFaceH4/ultrachat_200k"

    def pre_load_callback(self, split: str) -> str:
        split_map = {"train": "train_sft", "test": "test_sft"}
        return split_map.get(split, split)

    def post_load_callback(self, dataset: Dataset) -> Dataset:
        return dataset.select_columns(["messages"])
