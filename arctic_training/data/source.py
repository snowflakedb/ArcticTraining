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
from typing import Tuple

from datasets import disable_caching
from datasets import load_from_disk

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.data import DataSourceConfig
from arctic_training.data.factory import DataFactory
from arctic_training.data.utils import DatasetType
from arctic_training.data.utils import calculate_hash_from_args
from arctic_training.logging import logger
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method
from arctic_training.trainer.trainer import Trainer


class DataSource(ABC, CallbackMixin, metaclass=RegistryMeta):
    """Base DataSource class for loading training and evaluation data."""

    name: str
    """ Name of the DataSource. """

    config: DataSourceConfig
    """
    The type of the DataSourceConfig object that this DataSource uses. Any
    DataSource-specific options should be specified in this class.
    """

    @classmethod
    def _validate_subclass(cls) -> None:
        _validate_class_attribute_set(cls, "name")
        _validate_class_attribute_type(cls, "config", DataSourceConfig)
        _validate_class_method(cls, "load", ["self", "config", "split"])

    def __init__(self, data_factory: DataFactory, config: DataSourceConfig) -> None:
        self._data_factory = data_factory
        self.config = config

    def __call__(self, split: str) -> DatasetType:
        disable_caching()
        cache_path = self.cache_path(split)
        if self.data_factory.config.use_data_cache and cache_path.exists():
            logger.info(f"Loading data source from cache path {cache_path.as_posix()}")
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

        if self.data_factory.config.use_data_cache:
            dataset.save_to_disk(cache_path.as_posix())

        return dataset

    @property
    def trainer(self) -> Trainer:
        return self.data_factory.trainer

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
    def cache_path_args(self) -> Tuple[Dict, ...]:
        """Returns a dictionary of config fields that affect the cache path calculation."""

        # Some fields in the DataConfig should not affect cache path:
        # - sources / eval_sources: these are captures in the data source cache args
        # - cache_dir: this is the root of the cache path
        # - num_proc: does not affect output data
        # - train_eval_split: this is used after data is loaded/cached
        # - use_data_cache: does not affect the output data
        exclude_fields = [
            "sources",
            "eval_sources",
            "cache_dir",
            "num_proc",
            "train_eval_split",
            "use_data_cache",
        ]
        cache_path_args = (
            {
                k: v
                for k, v in self.data_factory.config.model_dump().items()
                if k not in exclude_fields
            },
            self.config.model_dump(),
            self.trainer.config.tokenizer.model_dump(),
        )
        return cache_path_args

    def cache_path(self, split: str) -> Path:
        """Returns the cache path for the data source split."""
        hash_str = calculate_hash_from_args(split, *self.cache_path_args)
        return self.data_factory.config.cache_dir / hash_str

    @callback_wrapper("load")
    @abstractmethod
    def load(self, config: DataSourceConfig, split: str) -> DatasetType:
        """Method to load the data. It should return a datasets.Dataset or datasets.IterableDataset."""
        raise NotImplementedError("load must be implemented in subclass")
