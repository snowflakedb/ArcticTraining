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
from pathlib import Path
from typing import TYPE_CHECKING
from typing import List
from typing import Optional
from typing import Tuple

import torch
from datasets import concatenate_datasets
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.data import DataConfig
from arctic_training.data.utils import DatasetType
from arctic_training.data.utils import calculate_hash_from_args
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method

if TYPE_CHECKING:
    from arctic_training.data.source import DataSource
    from arctic_training.trainer.trainer import Trainer


class DataFactory(ABC, CallbackMixin, metaclass=RegistryMeta):
    """Base DataFactory class for loading training and evaluation data."""

    name: str
    """
    Name of the DataFactory. This name should be unique to each registered
    DataFactory object. This name can be used in the training recipe YAMLs to
    specify the DataFactory to use.
    """

    config: DataConfig
    """
    The type of the DataConfig object that this DataFactory uses. Any
    DataFactory-specific options should be specified in this class.
    """

    @classmethod
    def _validate_subclass(cls) -> None:
        _validate_class_attribute_set(cls, "name")
        _validate_class_attribute_type(cls, "config", DataConfig)
        _validate_class_method(cls, "load", ["self", "data_sources", "split"])
        _validate_class_method(cls, "process", ["self", "dataset"])
        _validate_class_method(cls, "split_data", ["self", "training_data"])
        _validate_class_method(cls, "create_dataloader", ["self", "dataset"])

    def __init__(self, trainer: "Trainer", config: Optional[DataConfig] = None) -> None:
        if config is None:
            config = trainer.config.data

        self._trainer = trainer
        self.config = config

    def __call__(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        def get_data_split(split: str) -> Optional[DatasetType]:
            data_sources = self._get_data_sources(split=split)

            cache_path = self.cache_path(sources=data_sources, split=split)
            if self.config.use_data_cache and cache_path.exists():
                return load_from_disk(cache_path.as_posix())

            if len(data_sources) == 0:
                return None
            dataset = self.load(data_sources, split=split)
            dataset = self._truncate_data(dataset)

            if self.config.use_data_cache:
                dataset.save_to_disk(cache_path.as_posix())
            return dataset

        training_data = get_data_split("train")
        evaluation_data = get_data_split("eval")

        if self.config.train_eval_split[1] > 0.0:
            training_data, evaluation_data = self.split_data(training_data)

        training_dataloader = self.create_dataloader(training_data)
        evaluation_dataloader = (
            self.create_dataloader(evaluation_data)
            if evaluation_data is not None
            else None
        )

        return training_dataloader, evaluation_dataloader

    @property
    def trainer(self) -> "Trainer":
        """The Trainer object that is using this DataFactory."""
        return self._trainer

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The tokenizer object used by the Trainer."""
        return self.trainer.tokenizer

    @property
    def micro_batch_size(self) -> int:
        """The micro batch size used by the Trainer."""
        return self.trainer.config.micro_batch_size

    @property
    def global_rank(self) -> int:
        """The global rank of the current process."""
        return self.config.global_rank

    @property
    def world_size(self) -> int:
        """The total number of processes in the world."""
        return self.config.world_size

    def _get_data_sources(self, split: str) -> List["DataSource"]:
        if split == "train":
            data_source_configs = self.config.sources
        elif split == "eval":
            data_source_configs = self.config.eval_sources
        else:
            raise ValueError(f"Invalid split: {split}")

        data_sources = []
        for config in data_source_configs:
            data_source = config.data_source(data_factory=self, config=config)
            data_sources.append(data_source)
        return data_sources

    def _truncate_data(self, dataset: DatasetType) -> DatasetType:
        """
        Truncate the dataset to the shortest length across all processes.
        This ensures that each shard/process has the same number of samples in
        the dataset.
        """
        local_length = len(dataset)
        if self.world_size > 1:
            data_length = torch.zeros(self.world_size).to(self.trainer.device)
            data_length[self.global_rank] = local_length
            torch.distributed.all_reduce(data_length, op=torch.distributed.ReduceOp.SUM)
            shortest_length = int(data_length.min().item())
            del data_length  # clean the memory
        else:
            shortest_length = local_length
        dataset = dataset.select(range(shortest_length))
        return dataset

    def cache_path(self, sources: List["DataSource"], split: str) -> Path:
        """Returns the cache path for the processed + concatenated dataset."""
        source_cache_path_args = (s.cache_path_args for s in sources)
        hash_str = calculate_hash_from_args(split, *source_cache_path_args)
        return self.config.cache_dir / hash_str

    @callback_wrapper("load")
    def load(self, data_sources: List["DataSource"], split: str) -> DatasetType:
        """Loads data from one or more data sources and concatenates into a single dataset."""
        datasets = []
        for data_source in data_sources:
            dataset = data_source(split)
            datasets.append(dataset)
        dataset = concatenate_datasets(datasets)
        return dataset

    @callback_wrapper("process")
    def process(self, dataset: DatasetType) -> DatasetType:
        """Process the dataset (e.g., tokenization for text data)."""
        raise NotImplementedError(
            "tokenize must be implemented by DataFactory subclass."
        )

    @callback_wrapper("split")
    def split_data(
        self, training_data: DatasetType
    ) -> Tuple[DatasetType, Optional[DatasetType]]:
        """Split the training data into training and evaluation datasets."""
        datasets = training_data.train_test_split(
            test_size=self.config.train_eval_split[1],
            seed=self.config.seed,
        )
        training_data = datasets["train"]
        evaluation_data = datasets["test"]
        del datasets

        return training_data, evaluation_data

    @callback_wrapper("create_dataloader")
    def create_dataloader(self, dataset: DatasetType) -> DataLoader:
        """Create a torch DataLoader from the dataset."""
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            sampler=RandomSampler(dataset),
            num_workers=self.config.num_proc,
            drop_last=True,
        )
