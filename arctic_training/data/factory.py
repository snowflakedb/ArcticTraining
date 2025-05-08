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
from typing import Mapping
from typing import Optional
from typing import Tuple

import deepspeed.comm as dist
from datasets import concatenate_datasets
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from transformers import PreTrainedTokenizerBase

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.data import DataConfig
from arctic_training.config.data import DataSourceConfig
from arctic_training.data.utils import DatasetType
from arctic_training.data.utils import calculate_hash_from_args
from arctic_training.logging import logger
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
        _validate_class_method(cls, "load", ["self", "data_sources"])
        _validate_class_method(cls, "process", ["self", "dataset"])
        _validate_class_method(cls, "split_data", ["self", "training_data"])
        _validate_class_method(cls, "create_dataloader", ["self", "dataset"])

    def __init__(self, trainer: "Trainer", config: Optional[DataConfig] = None) -> None:
        if config is None:
            config = trainer.config.data

        self._trainer = trainer
        self.config = config

    def __call__(self) -> Tuple[DataLoader, Optional[Mapping[str, DataLoader]]]:
        def load_data_sources(
            data_source_configs: List[DataSourceConfig],
        ) -> Optional[DatasetType]:
            if len(data_source_configs) == 0:
                return None

            data_sources = self._get_data_sources(data_source_configs)
            cache_path = self.cache_path(sources=data_sources)

            # If the cache path does not exist, load the data using local/global
            # rank 0 (depending on if file system is shared across nodes).
            if self.is_main_process_by_path and not cache_path.exists():
                dataset = self.load(data_sources)

                # Repeat the dataset until we have enough samples to run for min_iterations
                if self.trainer.config.min_iterations > 0:
                    required_samples = (
                        self.trainer.config.min_iterations
                        * self.trainer.config.micro_batch_size
                        * self.trainer.config.gradient_accumulation_steps
                        * self.world_size
                    )
                    if required_samples > len(dataset):
                        num_repeats = required_samples // len(dataset) + 1
                        dataset = concatenate_datasets([dataset] * num_repeats)
                        dataset = dataset.select(range(required_samples))

                if len(dataset) < self.world_size:
                    raise ValueError(
                        f"Dataset size ({len(dataset)}) is less than the data parallel"
                        f" size ({self.world_size}) so not every rank will get a data"
                        " sample. For development and debugging work, you can set the"
                        " `min_iterations` parameter in the training config to"
                        " replicate the loaded data until there is enough data samples"
                        " to run for that many iterations."
                    )

                logger.info(f"Saving dataset to cache path {cache_path.as_posix()}")
                tmp_cache_path = cache_path.with_suffix(".incomplete")
                dataset.save_to_disk(tmp_cache_path.as_posix())
                tmp_cache_path.rename(cache_path)

            dist.barrier()  # Wait for the main process to finish its preprocessing + saving to cache

            # Reset seeds after may be processing data if cache didn't exist - so that main process ends up with the same RNG if the cache was there and if it wasn't, thus ensuring reproducibility.
            self.trainer._set_seeds(self.trainer.config.seed)
            logger.info(f"Loading dataset from cache path {cache_path.as_posix()}")
            return load_from_disk(cache_path.as_posix())

        training_data = load_data_sources(self.config.sources)
        evaluation_data = load_data_sources(self.config.eval_sources)

        if self.config.train_eval_split[1] > 0.0:
            training_data, evaluation_data = self.split_data(training_data)

        training_dataloader = self.create_dataloader(training_data)
        evaluation_dataloader = self.create_dataloader(evaluation_data) if evaluation_data is not None else None

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
    def local_rank(self) -> int:
        """The local rank of the current process."""
        return self.config.local_rank

    @property
    def world_size(self) -> int:
        """The total number of processes in the world."""
        return self.config.world_size

    def _get_data_sources(self, data_source_configs: List[DataSourceConfig]) -> List["DataSource"]:
        data_sources = []
        for config in data_source_configs:
            data_source = config.data_source(data_factory=self, config=config)
            data_sources.append(data_source)
        return data_sources

    @property
    def is_main_process_by_path(self) -> bool:
        if self.config.cache_fs_type == "local":
            return self.local_rank == 0
        return self.global_rank == 0

    def cache_path(self, sources: List["DataSource"]) -> Path:
        """Returns the cache path for the processed + concatenated dataset."""
        source_cache_path_args = (s.cache_path_args for s in sources)
        hash_str = calculate_hash_from_args(*source_cache_path_args)
        return self.config.cache_dir / hash_str

    @callback_wrapper("load")
    def load(self, data_sources: List["DataSource"]) -> DatasetType:
        """Loads data from one or more data sources and concatenates into a single dataset."""
        datasets = []
        for data_source in data_sources:
            dataset = data_source()
            datasets.append(dataset)
        dataset = concatenate_datasets(datasets)
        return dataset

    @callback_wrapper("process")
    def process(self, dataset: DatasetType) -> DatasetType:
        """Process the dataset (e.g., tokenization for text data)."""
        raise NotImplementedError("tokenize must be implemented by DataFactory subclass.")

    @callback_wrapper("split")
    def split_data(self, training_data: DatasetType) -> Tuple[DatasetType, Optional[DatasetType]]:
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
            sampler=DistributedSampler(dataset, num_replicas=self.world_size, rank=self.global_rank),
            num_workers=self.config.dl_num_workers,
            drop_last=True,
        )
