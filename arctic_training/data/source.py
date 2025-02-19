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

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import Optional

from datasets import disable_caching
from datasets import load_from_disk

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.data import DataSourceConfig
from arctic_training.data.factory import DataFactory
from arctic_training.data.utils import DatasetType


class DataSource(ABC, CallbackMixin):
    """Base DataSource class for loading training and evaluation data."""

    name: str
    """ Name of the DataSource. """

    config: DataSourceConfig
    """
    The type of the DataSourceConfig object that this DataSource uses. Any
    DataSource-specific options should be specified in this class.
    """

    def __init__(self, data_factory: DataFactory, config: DataSourceConfig) -> None:
        self._data_factory = data_factory
        self.config = config

    def __call__(self, split: str, cache_path: Optional[Path] = None) -> DatasetType:
        disable_caching()
        if cache_path is not None and cache_path.exists():
            return load_from_disk(cache_path.as_posix())

        dataset = self.load(self.config, split)
        if len(dataset) < 1:
            raise ValueError(
                f"Empty dataset from load() for data source type {self.name} with"
                f" config {self.config} for split {split}"
            )
        if self.config.shard:
            if len(dataset) < self.world_size:
                raise ValueError(
                    "Sharding is enabled but the dataset size is smaller than the"
                    f" number of shards. Dataset size: {len(dataset)}, number of"
                    f" shards: {self.world_size}"
                )
            dataset = dataset.shard(num_shards=self.world_size, index=self.global_rank)
        if self.config.process:
            dataset = self.data_factory.process(dataset)
            if len(dataset) < 1:
                raise ValueError(
                    "Empty dataset after process() for data source type"
                    f" {self.name} with config {self.config} for split {split}"
                )

        if cache_path is not None:
            dataset.save_to_disk(cache_path.as_posix())

        return dataset

    @property
    def data_factory(self) -> DataFactory:
        return self._data_factory

    @property
    def world_size(self) -> int:
        return self.data_factory.world_size

    @property
    def global_rank(self) -> int:
        return self.data_factory.global_rank

    @property
    def cache_path_args(self) -> Dict:
        return self.config.model_dump()

    @callback_wrapper("load")
    @abstractmethod
    def load(self, config: DataSourceConfig, split: str) -> DatasetType:
        """Method to load the data. It should return a tokenized HuggingFace Dataset or IterableDataset."""
        raise NotImplementedError("load must be implemented in subclass")
