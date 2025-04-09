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

from os.path import basename
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import fsspec
from torch.utils.data import DataLoader

from arctic_training.config.data import DataConfig
from arctic_training.config.data import DataSourceConfig
from arctic_training.data.factory import DataFactory
from arctic_training.logging import logger

from .core.pretokenized_batch_loader import ContrastiveLearningBatch
from .core.pretokenized_batch_loader import ContrastiveLearningBatchDataset

FilesystemOption = Literal["local", "s3"]


class ContrastivePretokenizedDataConfig(DataConfig):
    type: str = "contrastive_pretokenized"
    filesystem: FilesystemOption
    root_directory: str
    split_factor: int = 1
    sources: List[DataSourceConfig] = []
    max_seq_length_query: Optional[int] = None
    max_seq_length_doc: Optional[int] = None
    eval_root_directories: Optional[List[str]] = None
    eval_split_factor: int = 1
    eval_max_seq_length_query: Optional[int] = None
    eval_max_seq_length_doc: Optional[int] = None


class ContrastivePretokenizedDataFactory(DataFactory):
    name: str = "contrastive_pretokenized"
    config: ContrastivePretokenizedDataConfig

    def __call__(self) -> Tuple[DataLoader, Optional[Dict[str, DataLoader]]]:
        fs = self.get_filesystem(self.config.filesystem)

        # Create the train loader.
        train_ds = ContrastiveLearningBatchDataset(
            filesystem=fs,
            root_directory=self.config.root_directory,
            split_factor=self.config.split_factor,
            shard_id=self.global_rank,
            world_size=self.world_size,
            max_seq_len_query=self.config.max_seq_length_query,
            max_seq_len_doc=self.config.max_seq_length_doc,
            device=self.trainer.device,
        )
        train_dl = DataLoader(train_ds, batch_size=None)

        # Create the eval loaders.
        eval_dl_map: Dict[str, DataLoader[ContrastiveLearningBatch]] = {}
        if self.config.eval_root_directories is not None:
            for eval_root_directory in self.config.eval_root_directories:
                eval_name = basename(eval_root_directory)
                eval_ds = ContrastiveLearningBatchDataset(
                    filesystem=fs,
                    root_directory=eval_root_directory,
                    split_factor=self.config.eval_split_factor,
                    shard_id=self.global_rank,
                    world_size=self.world_size,
                    max_seq_len_query=self.config.eval_max_seq_length_query,
                    max_seq_len_doc=self.config.eval_max_seq_length_doc,
                    device=self.trainer.device,
                )
                eval_dl = DataLoader(eval_ds, batch_size=None)
                if eval_name in eval_dl_map:
                    raise ValueError(
                        f"Duplicate eval dataset name: {eval_name}. Each eval dataset must have a unique name."
                    )
                eval_dl_map[eval_name] = eval_dl
                logger.info(f"Added eval dataset {eval_name}. {len(eval_ds)=}")

        # Create the loaders
        # (NOTE: multiple eval datasets requires a different approach).

        return train_dl, eval_dl_map

    def get_eval_loaders(self) -> Dict[str, DataLoader]:
        assert hasattr(self, "eval_datasets"), "Must call the factory first"
        return {name: DataLoader(ds, batch_size=None) for name, ds in self.eval_datasets.items()}

    def tokenize(self, tokenizer, dataset):
        # No-op, assuming dataset already tokenized.
        return dataset

    @staticmethod
    def get_filesystem(filesystem_name: FilesystemOption):
        if filesystem_name == "local":
            return fsspec.filesystem("file")
        elif filesystem_name == "s3":
            larger_blocksize = 64 * (1024**2)  # 64 MiB vs. default of 5MiB
            return fsspec.filesystem("s3", use_listings_cache=False, default_block_size=larger_blocksize)
        else:
            raise ValueError(f"Unknown filesystem option: {filesystem_name}")
