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
from typing import TYPE_CHECKING
from typing import Dict
from typing import Generic
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
from arctic_training.data.utils import DatasetType
from arctic_training.data.utils import calculate_hash_from_args
from arctic_training.registry.data import get_registered_data_source

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer
    from arctic_training.data.source import DataSource

from arctic_training.config.data import TDataConfig


class DataFactory(ABC, CallbackMixin, Generic[TDataConfig]):
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

    def __init__(self, trainer: "Trainer", config: Optional[DataConfig] = None) -> None:
        if config is None:
            config = trainer.config.data

        self._trainer = trainer
        self.config = config

    def __call__(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        datasets: Dict[str, DatasetType] = {}
        for split in ("train", "eval"):
            data_sources = self._get_data_sources(split=split)
            if not data_sources:
                datasets[split] = None
                continue
            cache_path = self._get_processed_data_cache_path(data_sources)
            if self.config.use_data_cache and cache_path.exists():
                dataset = load_from_disk(cache_path.as_posix())
            else:
                dataset = self.load(data_sources, split=split)
                dataset = self._truncate_data(dataset)
                if self.config.use_data_cache:
                    dataset.save_to_disk(cache_path.as_posix())
            datasets[split] = dataset

        training_data, evaluation_data = datasets["train"], datasets["eval"]

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
        return self.trainer.global_rank

    @property
    def world_size(self) -> int:
        """The total number of processes in the world."""
        return self.trainer.world_size

    @property
    def cache_path_args(self) -> Dict:
        cache_path_args = {}
        for field in self.config.model_fields.keys():
            if field not in ["sources", "eval_sources"]:
                cache_path_args[field] = getattr(self.config, field)
        return cache_path_args

    def _get_source_cache_path(self, source: "DataSource") -> Path:
        hash_str = calculate_hash_from_args(
            self.cache_path_args, source.cache_path_args
        )
        return self.config.cache_dir / hash_str

    def _get_processed_data_cache_path(
        self, data_source_list: List["DataSource"]
    ) -> Path:
        hash_str = calculate_hash_from_args(
            self.cache_path_args,
            *[source.cache_path_args for source in data_source_list],
        )
        return self.config.cache_dir / hash_str

    def _get_data_sources(self, split: str) -> List["DataSource"]:
        if split == "train":
            data_source_configs = self.config.sources
        elif split == "eval":
            data_source_configs = self.config.eval_sources
        else:
            raise ValueError(f"Invalid split: {split}")

        data_sources = []
        for config in data_source_configs:
            data_source_cls = get_registered_data_source(name_or_class=config.type)
            data_sources.append(data_source_cls(self, config))

        return data_sources

    def _truncate_data(self, dataset: DatasetType) -> DatasetType:
        local_length = len(dataset)
        if self.world_size != 1:
            data_length = torch.zeros(self.world_size).cuda()
            data_length[self.global_rank] = local_length
            torch.distributed.all_reduce(data_length, op=torch.distributed.ReduceOp.SUM)
            shortest_length = data_length.min().cpu().item()
            del data_length  # clean the memory
        else:
            shortest_length = local_length
        dataset = dataset.select(range(shortest_length))
        return dataset

    @callback_wrapper("load")
    def load(self, data_sources: List["DataSource"], split: str) -> DatasetType:
        datasets = []
        for data_source in data_sources:
            cache_path = self._get_source_cache_path(data_source)
            dataset = data_source(split, cache_path=cache_path)
            datasets.append(dataset)
        dataset = concatenate_datasets(datasets)
        return dataset

    @callback_wrapper("tokenize")
    @abstractmethod
    def tokenize(
        self, tokenizer: PreTrainedTokenizerBase, dataset: DatasetType
    ) -> DatasetType:
        raise NotImplementedError(
            "tokenize must be implemented by DataFactory subclass."
        )

    @callback_wrapper("split")
    def split_data(
        self, training_data: DatasetType
    ) -> Tuple[DatasetType, Optional[DatasetType]]:
        tmp = training_data.train_test_split(
            test_size=self.config.train_eval_split[1],
            seed=self.config.seed,
        )
        training_data = tmp["train"]
        evaluation_data = tmp["test"]
        del tmp

        return training_data, evaluation_data

    @callback_wrapper("create_dataloader")
    def create_dataloader(self, dataset: DatasetType) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.micro_batch_size,
            sampler=RandomSampler(dataset),
            num_workers=self.config.num_proc,
            drop_last=True,
        )
