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

if TYPE_CHECKING:
    from arctic_training.optimizer.factory import OptimizerFactory

from arctic_training.registry.utils import AlreadyRegisteredError
from arctic_training.registry.utils import _validate_class_attribute_set
from arctic_training.registry.utils import _validate_method_definition

_supported_optimizer_factory_registry: Dict[str, Type["OptimizerFactory"]] = {}


def register_optimizer_factory(
    cls: Type["OptimizerFactory"], force: bool = False
) -> Type["OptimizerFactory"]:
    global _supported_optimizer_factory_registry
    from arctic_training.optimizer.factory import OptimizerFactory

    if not issubclass(cls, OptimizerFactory):
        raise ValueError(
            f"New Optimizer Factory {cls.__name__} class must be a subclass of"
            " OptimzerFactory."
        )

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_set(cls, "config_type")
    _validate_method_definition(
        cls, "create_optimizer", ["self", "model", "optimizer_config"]
    )

    if cls.name in _supported_optimizer_factory_registry:
        if not force:
            raise AlreadyRegisteredError(
                f"Optimizer Factory {cls.name} is already registered. If you want ot"
                " overwrite, set force=True"
            )
        logger.warning(f"Overwriting existing registered Optimizer Factory {cls.name}.")

    _supported_optimizer_factory_registry[cls.name] = cls
    logger.info(f"Registered OptimizerFactory {cls.name}")

    return cls


def get_registered_optimizer_factory(
    name_or_class: Union[str, Type["OptimizerFactory"]],
) -> Type["OptimizerFactory"]:
    global _supported_optimizer_factory_registry

    if isinstance(name_or_class, str):
        optimizer_factory_name = name_or_class
    else:
        optimizer_factory_name = name_or_class.name

    if optimizer_factory_name not in _supported_optimizer_factory_registry:
        raise ValueError(
            f"{optimizer_factory_name} is not registered Optimizer Factory."
        )

    return _supported_optimizer_factory_registry[optimizer_factory_name]
