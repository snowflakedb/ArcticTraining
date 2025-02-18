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

from typing import TYPE_CHECKING
from typing import Dict
from typing import Type
from typing import Union

if TYPE_CHECKING:
    from arctic_training.data.factory import DataFactory
    from arctic_training.data.source import DataSource

from arctic_training.logging import logger
from arctic_training.registry.utils import AlreadyRegisteredError
from arctic_training.registry.utils import _validate_class_attribute_set
from arctic_training.registry.utils import _validate_class_attribute_type
from arctic_training.registry.utils import _validate_method_definition

_supported_data_source_registry: Dict[str, Type["DataSource"]] = {}
_supported_data_factory_registry: Dict[str, Type["DataFactory"]] = {}


def register_data_source(
    cls: Type["DataSource"], force: bool = False
) -> Type["DataSource"]:
    from arctic_training.config.data import DataSourceConfig

    global _supported_data_source_registry

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_type(cls, "config", DataSourceConfig)
    _validate_method_definition(cls, "load", ["self", "config", "split"])

    if cls.name in _supported_data_source_registry and not force:
        raise AlreadyRegisteredError(
            f"DataSource {cls.name} with type is already registered. If you want to"
            " overwrite, set force=True."
        )

    _supported_data_source_registry[cls.name] = cls
    logger.info(f"Registered DataSource {cls.name}.")

    return cls


def register_data_factory(
    cls: Type["DataFactory"], force: bool = False
) -> Type["DataFactory"]:
    from arctic_training.config.data import DataConfig

    global _supported_data_factory_registry

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_type(cls, "config", DataConfig)
    _validate_method_definition(cls, "load", ["self", "data_sources", "split"])
    _validate_method_definition(cls, "tokenize", ["self", "tokenizer", "dataset"])
    _validate_method_definition(cls, "split_data", ["self", "training_data"])
    _validate_method_definition(cls, "create_dataloader", ["self", "dataset"])

    if cls.name in _supported_data_factory_registry:
        if cls == _supported_data_factory_registry[cls.name]:
            return cls
        raise ValueError(
            f"DataFactory {cls.name} is already registered. If you want to overwrite,"
            " set force=True."
        )

    _supported_data_factory_registry[cls.name] = cls
    logger.info(f"Registered DataFactory {cls.name}.")

    return cls


def get_registered_data_source(
    name_or_class: Union[str, Type["DataSource"]],
) -> Type["DataSource"]:
    global _supported_data_source_registry

    if isinstance(name_or_class, str):
        data_source_name = name_or_class
    else:
        data_source_name = name_or_class.name

    if data_source_name not in _supported_data_source_registry:
        raise ValueError(f"Data source {data_source_name} is not registered.")

    return _supported_data_source_registry[data_source_name]


def get_registered_data_factory(
    name_or_class: Union[str, Type["DataFactory"]],
) -> Type["DataFactory"]:
    global _supported_data_factory_registry

    if isinstance(name_or_class, str):
        data_factory_name = name_or_class
    else:
        data_factory_name = name_or_class.name

    if data_factory_name not in _supported_data_factory_registry:
        raise ValueError(f"{data_factory_name} is not registered Data Factory.")

    return _supported_data_factory_registry[data_factory_name]
