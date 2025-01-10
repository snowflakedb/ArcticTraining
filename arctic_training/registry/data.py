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

from collections import defaultdict
from typing import TYPE_CHECKING
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

if TYPE_CHECKING:
    from arctic_training.data.factory import DataFactory
    from arctic_training.data.source import DataSource

from arctic_training.logging import logger
from arctic_training.registry.utils import AlreadyRegisteredError
from arctic_training.registry.utils import _validate_class_attribute_set
from arctic_training.registry.utils import _validate_method_definition

_supported_data_source_registry: Dict[str, Dict[str, Type["DataSource"]]] = defaultdict(
    dict
)
_supported_data_factory_registry: Dict[str, Type["DataFactory"]] = {}


def register_data_source(
    cls: Type["DataSource"], force: bool = False
) -> Type["DataSource"]:
    global _supported_data_source_registry

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_set(cls, "data_factory_type")
    _validate_method_definition(cls, "load_fn", ["self", "num_proc", "eval"])

    if isinstance(cls.data_factory_type, str):
        data_factory_type = cls.data_factory_type
    else:
        data_factory_type = cls.data_factory_type.name

    if cls.name in _supported_data_source_registry[data_factory_type] and not force:
        raise AlreadyRegisteredError(
            f"DataSource {cls.name} with type {data_factory_type} is already"
            " registered. If you want to overwrite, set force=True."
        )

    _supported_data_source_registry[data_factory_type][cls.name] = cls
    logger.info(f"Registered DataSource {cls.name} for {data_factory_type}.")

    return cls


def register_data_factory(
    cls: Type["DataFactory"], force: bool = False
) -> Type["DataFactory"]:
    global _supported_data_factory_registry

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_set(cls, "config_type")
    _validate_method_definition(cls, "load", ["self", "num_proc", "eval"])

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
    data_factory_type: Optional[str],
) -> Type["DataSource"]:
    global _supported_data_source_registry

    if isinstance(name_or_class, str):
        if not data_factory_type:
            raise ValueError(
                f"Must provide data_factory_type for data source {name_or_class}."
            )
        data_source_name = name_or_class
    else:
        data_source_name = name_or_class.name
        if (
            data_factory_type is not None
            and data_factory_type != name_or_class.data_factory_type
        ):
            raise ValueError(
                f"Provided data_factory_type ({data_factory_type}) does not match the"
                f" data source type ({name_or_class.data_factory_type})."
            )
        data_factory_type = name_or_class.data_factory_type

    if data_source_name not in _supported_data_source_registry[data_factory_type]:
        raise ValueError(
            f"Data source {data_source_name} for data factory type"
            f" {data_factory_type} is not registered."
        )

    return _supported_data_source_registry[data_factory_type][data_source_name]


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
