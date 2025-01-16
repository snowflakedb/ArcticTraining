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
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type

from pydantic import model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from arctic_training.data.factory import DataFactory

from arctic_training.logging import logger
from arctic_training.registry.data import get_registered_data_factory

from .base import BaseConfig


class DataConfig(BaseConfig):
    type: str = ""
    """ Data factory type. Defaults to the `data_factory_type` in the trainer. """

    sources: List[str]
    """ List of data sources to use for training. These must be registered `DataSource`. """

    eval_sources: List[str] = []
    """ list of data sources to use for evaluation. These must be registered `DataSource`. """

    train_eval_split: Tuple[float, float] = (1.0, 0.0)
    """ How much of the training data to use for evaluation. """

    max_length: int = 8192
    """ Maximum length of the input sequence. """

    num_proc: int = 16
    """ Number of processes to use for data loading. """

    seed: int = 42
    """ Seed for data loading. """

    mask_inputs: bool = True
    """ Whether to mask the input sequence. """

    use_data_cache: bool = True
    """ Whether to cache loaded data. """

    cache_processed_data: bool = False
    """ Whether to cache processed data. """

    cache_dir: Path = Path("/tmp/")
    """ Directory to store cached data. """

    always_max_length: bool = False
    """
    If this is turned on, each batch will be filled up to the max length by
    appending samples until the total length matches the max length. It might
    cause the last sample to be truncated.
    """

    @property
    def cache_path_args(self) -> Dict[str, Any]:
        """Returns the fields that are used to generate the cache path."""
        cache_path_args = {}
        for field in self.model_fields.keys():
            if field not in ["cache_dir", "num_proc", "sources", "eval_sources"]:
                cache_path_args[field] = getattr(self, field)
        return cache_path_args

    @property
    def factory(self) -> Type["DataFactory"]:
        return get_registered_data_factory(self.type)

    @model_validator(mode="after")
    def validate_cache_dir(self) -> Self:
        if self.use_data_cache:
            assert (
                self.cache_dir is not None
            ), "You must provide a data_cache_dir if use_data_cache is True."
            if not self.cache_dir.exists():
                logger.warning(
                    f"Caching directory {self.cache_dir} does not exist. Creating it."
                )
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def validate_train_eval_split(self) -> Self:
        if self.eval_sources:
            assert (
                self.train_eval_split[0] == 1.0
            ), "train_eval_split should be (1.0, 0.0) when eval_datasets is provided."
        if self.train_eval_split[1] > 0.0:
            assert not self.eval_sources, (
                "If you provide the evaluation split, you should not provide the"
                " evaluation datasets."
            )
        assert sum(self.train_eval_split) == 1.0, "train_eval_split should sum to 1.0."
        return self
