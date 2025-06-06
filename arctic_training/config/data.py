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
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from pydantic import ValidationInfo
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

from arctic_training.config.base import BaseConfig
from arctic_training.config.utils import HumanInt
from arctic_training.data.utils import is_local_fs
from arctic_training.exceptions import RegistryError
from arctic_training.logging import logger
from arctic_training.registry import _get_class_attr_type_hints
from arctic_training.registry import get_registered_data_factory
from arctic_training.registry import get_registered_data_source

if TYPE_CHECKING:
    from arctic_training.data.factory import DataFactory
    from arctic_training.data.source import DataSource


class DataSourceConfig(BaseConfig):
    """Base DataSource configuration."""

    type: str = ""
    """ Data source type. Defaults to 'huggingface' if only a dataset name or path is provided."""

    split: str = ""
    """
    Which split the data source is used for. This will be automatically set to either "train" or "eval" if no value is passed.

    For HFDataSource, this can be any value supported by Dataset slice splits:
    https://huggingface.co/docs/datasets/en/loading#slice-splits.
    """

    sample_ratio: Optional[float] = None
    """Ratio of the dataset to randomly sample. If None, all examples are used."""

    sample_count: Optional[int] = None
    """Number of examples to randomly sample. If None, all examples are used."""

    sample_seed: int = 42
    """Seed for random sampling. Used only if `sample_ratio` or `sample_count` is set."""

    process: bool = True
    """ Whether to process the data with the data factory `process` function (e.g., tokenization for SFTDataFactory). """

    @property
    def data_source(self) -> Type["DataSource"]:
        return get_registered_data_source(self.type)

    @model_validator(mode="after")
    def sample_ratio_or_sample_count(self) -> Self:
        assert (
            self.sample_ratio is None or self.sample_count is None
        ), "sample_ratio and sample_count cannot both be set."
        return self


class DataConfig(BaseConfig):
    type: str = ""
    """ Data factory type. Defaults to the `data_factory_type` in the trainer. """

    sources: List[DataSourceConfig]
    """ List of data sources to use for training. These must be registered `DataSource`. """

    eval_sources: List[DataSourceConfig] = []
    """ list of data sources to use for evaluation. These must be registered `DataSource`. """

    train_eval_split: Tuple[float, float] = (1.0, 0.0)
    """ How much of the training data to use for evaluation. """

    max_length: HumanInt = 8192
    """ Maximum length of the input sequence. """

    num_proc: int = 16
    """ Number of processes to use for data loading. """

    dl_num_workers: int = 2
    """ Number of DL workers per gpu. """

    seed: int = 42
    """ Seed for data loading. """

    use_data_cache: Optional[bool] = None
    """ Whether to cache loaded data. """

    cache_processed_data: Optional[bool] = None
    """ Deprecated, please use "use_data_cache". """

    cache_dir: Path = Path("/tmp/")
    """ Directory to store cached data. """

    cache_fs_type: Literal["auto", "local", "shared"] = "auto"

    @property
    def factory(self) -> Type["DataFactory"]:
        return get_registered_data_factory(self.type)

    @field_validator("use_data_cache", "cache_processed_data", mode="after")
    @classmethod
    def deprecate_cache_processed_data(cls, v: Optional[bool]) -> Optional[bool]:
        if v is not None:
            logger.warning(
                "The 'use_data_cache' and 'cache_processed_data' fields are deprecated."
                " Data cache is used by default now."
            )
        return v

    @field_validator("cache_dir", mode="after")
    @classmethod
    def resolve_cache_dir(cls, v: Path) -> Path:
        return v.resolve()

    @field_validator("sources", "eval_sources", mode="before")
    def init_source_configs(
        cls,
        v: List[Union[str, Dict, DataSourceConfig]],
        info: ValidationInfo,
    ) -> List[DataSourceConfig]:
        """Convert string and dict input to correct subclass of DataSourceConfig. If a string is passed, "huggingface" is used as the DataSource type."""
        data_configs = []
        for config in v:
            split = "train" if info.field_name == "sources" else "eval"

            # Support passing just a dataset name or path
            if isinstance(config, str):
                # User has passed split suffix as part of the name
                if ":" in config:
                    config, split = config.split(":", 1)

                try:
                    _ = get_registered_data_source(config)
                    config = dict(type=config, name_or_path=config)
                except RegistryError:
                    config = dict(type="huggingface", name_or_path=config)

            # Convert passed dictionary to DataSourceConfig subclass
            if isinstance(config, dict):
                if "type" not in config:
                    raise KeyError(
                        "Unspecified data source type. Please specify the 'type' field"
                        f" in your datasource config. Error raised for input: {config}."
                    )
                if "split" not in config:
                    config["split"] = split
                data_source_cls = get_registered_data_source(config["type"])
                config_cls = _get_class_attr_type_hints(data_source_cls, "config")[0]
                data_configs.append(config_cls(**config))
            else:
                data_configs.append(config)
        return data_configs

    @model_validator(mode="after")
    def validate_cache_dir(self) -> Self:
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def validate_train_eval_split(self) -> Self:
        if self.eval_sources:
            assert (
                self.train_eval_split[0] == 1.0
            ), "train_eval_split should be (1.0, 0.0) when eval_datasets is provided."
        if self.train_eval_split[1] > 0.0:
            assert (
                not self.eval_sources
            ), "If you provide the evaluation split, you should not provide the evaluation datasets."
        assert sum(self.train_eval_split) == 1.0, "train_eval_split should sum to 1.0."
        return self

    @model_validator(mode="after")
    def set_cache_fs_type(self) -> Self:
        if self.cache_fs_type == "auto":
            self.cache_fs_type = "local" if is_local_fs(self.cache_dir) else "shared"
        return self
