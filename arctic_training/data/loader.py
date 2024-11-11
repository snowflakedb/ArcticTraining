import hashlib
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import List

import torch
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import disable_caching
from datasets import load_from_disk
from transformers import PreTrainedTokenizerBase

from .sft_utils import packing_sft_dataset

if TYPE_CHECKING:
    from arctic_training.config import DataConfig

from arctic_training.register import get_dataset_class


class DataLoaderBase(ABC):
    @property
    def world_size(self) -> int:
        return torch.distributed.get_world_size()

    @property
    def global_rank(self) -> int:
        return torch.distributed.get_rank()

    def get_cache_path(self, cache_dir: Path, *args: Any) -> Path:
        # TODO: Fix the args that are passed in to account for changes in config
        # that do not affect data loading process (e.g., data cache dir)
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


class DataSetLoader(DataLoaderBase):
    dataset_name: str = ""
    dataset_type: str = ""

    def __init__(
        self,
        dataset: str,
        tokenizer: PreTrainedTokenizerBase,
        eval: bool,
        config: "DataConfig",
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.eval = eval
        self.config = config

    @abstractmethod
    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        raise NotImplementedError("load_fn method is not implemented.")

    @abstractmethod
    def tokenize_fn(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        data_config: "DataConfig",
    ) -> Any:
        raise NotImplementedError("tokenize_fn method is not implemented.")

    def load_dataset(self) -> Dataset:
        disable_caching()

        if self.config.use_data_cache and self.cache_path.exists():
            return load_from_disk(self.cache_path)

        dataset = self.load_fn(num_proc=self.config.num_proc, eval=self.eval)

        dataset = dataset.shard(num_shards=self.world_size, index=self.global_rank)
        dataset = dataset.select(range(len(dataset)))
        dataset = self.tokenize_fn(
            dataset=dataset, tokenizer=self.tokenizer, data_config=self.config
        )

        if self.config.use_data_cache:
            print(f"SAVING DATA: {self.cache_path}")
            dataset.save_to_disk(self.cache_path)

        return dataset

    @property
    def cache_path(self) -> Path:
        # TODO: fix this
        return self.get_cache_path(
            self.config.data_cache_dir, self.config, self.world_size, self.global_rank
        )


class ConcatenatedDataSetsLoader(DataLoaderBase):
    def __init__(
        self,
        dataset_list: List[str],
        tokenizer: PreTrainedTokenizerBase,
        eval: bool,
        config: "DataConfig",
    ) -> None:
        self.dataset_list: List[str] = dataset_list
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.eval: bool = eval
        self.config: "DataConfig" = config

    def load_datasets(self) -> Dataset:
        if self.config.use_data_cache and self.cache_path.exists():
            return load_from_disk(self.cache_path)

        datasets = [d.load_dataset() for d in self.dataset_loaders]
        dataset = concatenate_datasets(datasets)

        # TODO can this be moved to the DataSetLoader?
        if self.config.dataset_type == "sft":
            dataset = packing_sft_dataset(
                dataset,
                seed=self.config.seed,
                rank=self.global_rank,
                max_length=self.config.max_length,
            )

        truncate_length = self.get_shortest_data_length(dataset)
        dataset = dataset.select(range(truncate_length))

        if self.config.cache_processed_data:
            dataset.save_to_disk(self.cache_path)

        return dataset

    def get_shortest_data_length(self, dataset: Dataset) -> int:
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

    @property
    def dataset_loaders(self) -> List[DataSetLoader]:
        loaders = []
        for dataset in self.dataset_list:
            loader_cls = get_dataset_class(
                dataset_type=self.config.dataset_type, dataset_name=dataset
            )
            loaders.append(
                loader_cls(
                    dataset=dataset,
                    tokenizer=self.tokenizer,
                    eval=self.eval,
                    config=self.config,
                )
            )
        return loaders

    @property
    def cache_path(self) -> Path:
        # TODO: fix this
        return self.get_cache_path(
            self.config.data_cache_dir,
            self.dataset_list,
            self.config,
            self.world_size,
            self.global_rank,
        )
