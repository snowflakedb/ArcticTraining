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

import hashlib
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import torch
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import disable_caching
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from transformers import PreTrainedTokenizerBase

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.config.data import DataConfig
from arctic_training.registry.data import get_registered_data_source

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer


class DataFactory(ABC, CallbackMixin):
    """Base DataFactory class for loading training and evaluation data."""

    name: str
    """
    Name of the DataFactory. This name should be unique to each registered
    DataFactory object. This name can be used in the training recipe YAMLs to
    specify the DataFactory to use.
    """

    config_type: Type[DataConfig]
    """
    The type of the DataConfig object that this DataFactory uses. Any
    DataFactory-specific options should be specified in this class.
    """

    def __init__(
        self, trainer: "Trainer", data_config: Optional["DataConfig"] = None
    ) -> None:
        if data_config is None:
            data_config = trainer.config.data

        self._trainer = trainer
        self.config = data_config

    def __call__(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        train_dataset = self._load_data(
            data_source_list=self.config.sources, eval=False
        )
        eval_dataset = None

        if self.config.eval_sources:
            eval_dataset = self._load_data(
                data_source_list=self.config.eval_sources, eval=True
            )
        elif self.config.train_eval_split[1] > 0.0:
            tmp = train_dataset.train_test_split(
                test_size=self.config.train_eval_split[1],
                seed=self.config.seed,
            )
            train_dataset = tmp["train"]
            eval_dataset = tmp["test"]
            del tmp

        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=self.collate_fn(tokenizer=self.tokenizer),
            batch_size=self.micro_batch_size,
            sampler=RandomSampler(train_dataset),
            num_workers=self.config.num_proc,
            drop_last=True,
        )
        eval_dataloader = None
        if eval_dataset is not None:
            eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=self.collate_fn(tokenizer=self.tokenizer),
                batch_size=self.micro_batch_size,
                sampler=RandomSampler(eval_dataset),
                num_workers=self.config.num_proc,
                drop_last=True,
            )

        return train_dataloader, eval_dataloader

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

    def _load_data(self, data_source_list: List[str], eval: bool) -> Dataset:
        cache_path = self._processed_data_cache(data_source_list=data_source_list)
        if self.config.use_data_cache and cache_path.exists():
            return load_from_disk(cache_path.as_posix())

        datasets = [
            self._load_data_source(data_source, eval)
            for data_source in data_source_list
        ]
        dataset = concatenate_datasets(datasets)

        # TODO: better name for this method
        dataset = self.modify_dataset(dataset)

        truncate_length = self._get_shortest_data_length(dataset)
        dataset = dataset.select(range(truncate_length))

        if self.config.use_data_cache:
            dataset.save_to_disk(cache_path.as_posix())

        return dataset

    def _load_data_source(self, data_source: str, eval: bool) -> Dataset:
        disable_caching()
        cache_path = self._data_source_cache(data_source=data_source)
        if self.config.use_data_cache and cache_path.exists():
            dataset = load_from_disk(cache_path.as_posix())
            return load_from_disk(cache_path.as_posix())

        data_source_cls = get_registered_data_source(
            name_or_class=data_source, data_factory_type=self.name
        )
        data_source_loader = data_source_cls(num_proc=self.config.num_proc, eval=eval)
        dataset = data_source_loader()
        dataset = dataset.shard(num_shards=self.world_size, index=self.global_rank)
        dataset = self.tokenize_fn(trainer=self.trainer, dataset=dataset)

        if self.config.use_data_cache:
            dataset.save_to_disk(cache_path.as_posix())

        return dataset

    def _get_shortest_data_length(self, dataset: Dataset) -> int:
        local_length = len(dataset)
        data_length = torch.zeros(self.world_size).cuda()
        data_length[self.global_rank] = local_length
        try:
            torch.distributed.all_reduce(data_length, op=torch.distributed.ReduceOp.SUM)
        except Exception:
            # single GPU quick test
            pass
        shortest_length = data_length.min().cpu().item()
        del data_length  # clean the memory
        return int(shortest_length)

    def _get_cache_path(self, cache_dir: Path, *args: Any) -> Path:
        assert args, "At least one argument is required to generate cache path."
        hash_str = ""
        for arg in args:
            try:
                hash_str += str(arg)
            except Exception as e:
                raise ValueError(f"Failed to convert {arg} to string. Error: {e}")
        cache_name = hashlib.md5(hash_str.encode()).hexdigest()
        cache_path = cache_dir / cache_name
        return cache_path

    def _processed_data_cache(self, data_source_list: List[str]) -> Path:
        cache_dir = self.config.cache_dir
        return self._get_cache_path(
            cache_dir,
            data_source_list,
            self.config.cache_path_args,
            self.world_size,
            self.global_rank,
        )

    def _data_source_cache(self, data_source: str) -> Path:
        cache_dir = self.config.cache_dir
        return self._get_cache_path(
            cache_dir,
            data_source,
            self.config.cache_path_args,
            self.world_size,
            self.global_rank,
        )

    @abstractmethod
    def tokenize_fn(self, trainer: "Trainer", dataset: Dataset) -> Dataset:
        """
        Function to tokenize the dataset. This function should be implemented by
        the subclass and return a Dataset of tokenized data.
        """
        raise NotImplementedError(
            "tokenize_fn must be implemented by DataFactory subclass."
        )

    @staticmethod
    @abstractmethod
    def collate_fn(tokenizer: PreTrainedTokenizerBase) -> Callable:
        """
        Returns a collate function that can be used with a Torch DataLoader.
        """
        raise NotImplementedError(
            "collate_fn must be implemented by DataFactory subclass."
        )

    def modify_dataset(self, dataset: Dataset) -> Dataset:
        """
        Called after concatenating multiple data sources and before caching the
        processed data (if enabled).
        """
        return dataset
