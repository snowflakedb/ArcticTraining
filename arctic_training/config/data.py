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
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

from arctic_training.config.base import BaseConfig
from arctic_training.logging import logger
from arctic_training.registry.data import get_registered_data_factory
from arctic_training.registry.data import get_registered_data_source
from arctic_training.registry.utils import _get_class_attr_type_hints

if TYPE_CHECKING:
    from arctic_training.data.factory import DataFactory
    from arctic_training.data.source import DataSource


class DataSourceConfig(BaseConfig):
    """Base DataSource configuration."""

    type: str = ""
    """ Data source type. """

    process: bool = True
    """ Whether to process the data with the data factory `process` function (e.g., tokenization for SFTDataFactory). """

    shard: bool = True
    """ Whether to shard the data. """

    @property
    def data_source(self) -> Type["DataSource"]:
        return get_registered_data_source(self.type)


class DataConfig(BaseConfig):
    type: str = ""
    """ Data factory type. Defaults to the `data_factory_type` in the trainer. """

    sources: List[DataSourceConfig]
    """ List of data sources to use for training. These must be registered `DataSource`. """

    eval_sources: List[DataSourceConfig] = []
    """ list of data sources to use for evaluation. These must be registered `DataSource`. """

    train_eval_split: Tuple[float, float] = (1.0, 0.0)
    """ How much of the training data to use for evaluation. """

    num_proc: int = 16
    """ Number of processes to use for data loading. """

    seed: int = 42
    """ Seed for data loading. """

    use_data_cache: bool = True
    """ Whether to cache loaded data. """

    cache_processed_data: bool = False
    """ Whether to cache processed data. """

    cache_dir: Path = Path("/tmp/")
    """ Directory to store cached data. """

    @property
    def factory(self) -> Type["DataFactory"]:
        return get_registered_data_factory(self.type)

    @field_validator("sources", "eval_sources", mode="before")
    def create_source_config_from_name(
        cls, v: List[Union[str, Dict, DataSourceConfig]]
    ) -> List[DataSourceConfig]:
        data_sources = []
        for source in v:
            if isinstance(source, str):
                data_source_cls = get_registered_data_source(source)
                config_cls = _get_class_attr_type_hints(data_source_cls, "config")[0]
                data_sources.append(config_cls(type=source))
            else:
                data_sources.append(source)
        return data_sources

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
