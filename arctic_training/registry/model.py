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

from arctic_training.logging import logger
from arctic_training.registry.utils import AlreadyRegisteredError
from arctic_training.registry.utils import _validate_class_attribute_set
from arctic_training.registry.utils import _validate_class_attribute_type
from arctic_training.registry.utils import _validate_method_definition

if TYPE_CHECKING:
    from arctic_training.model.factory import ModelFactory

_supported_model_factory_registry: Dict[str, Type["ModelFactory"]] = {}


def register_model_factory(
    cls: Type["ModelFactory"], force: bool = False
) -> Type["ModelFactory"]:
    from arctic_training.config.model import ModelConfig
    from arctic_training.model.factory import ModelFactory

    global _supported_model_factory_registry

    if not issubclass(cls, ModelFactory):
        raise ValueError(
            f"New Model Factory {cls.__name__} class must be a subclass of"
            " ModelFactory."
        )

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_type(cls, "config", ModelConfig)
    _validate_method_definition(cls, "create_config", ["self"])
    _validate_method_definition(cls, "create_model", ["self", "model_config"])

    if cls.name in _supported_model_factory_registry:
        if not force:
            raise AlreadyRegisteredError(
                f"Model Factory {cls.name} is already registered. If you want ot"
                " overwrite, set force=True"
            )
        logger.warning(f"Overwriting existing registered Model Factory {cls.name}.")

    _supported_model_factory_registry[cls.name] = cls
    logger.info(f"Registered ModelFactory {cls.name}")

    return cls


def get_registered_model_factory(
    name_or_class: Union[str, Type["ModelFactory"]],
) -> Type["ModelFactory"]:
    global _supported_model_factory_registry

    if isinstance(name_or_class, str):
        model_factory_name = name_or_class
    else:
        model_factory_name = name_or_class.name

    if model_factory_name not in _supported_model_factory_registry:
        raise ValueError(
            f"{model_factory_name} is not a registered Model Factory! Registered Model"
            f" Factories are: {list(_supported_model_factory_registry.keys())}"
        )

    return _supported_model_factory_registry[model_factory_name]
