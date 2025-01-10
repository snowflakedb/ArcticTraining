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
from arctic_training.registry.utils import _validate_method_definition

if TYPE_CHECKING:
    from arctic_training.scheduler.factory import SchedulerFactory

_supported_scheduler_factory_registry: Dict[str, Type["SchedulerFactory"]] = {}


def register_scheduler_factory(
    cls: Type["SchedulerFactory"], force: bool = False
) -> Type["SchedulerFactory"]:
    global _supported_scheduler_factory_registry
    from arctic_training.scheduler.factory import SchedulerFactory

    if not issubclass(cls, SchedulerFactory):
        raise ValueError(
            f"New Scheduler Factory {cls.__name__} class must be a subclass of"
            " OptimzerFactory."
        )

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_set(cls, "config_type")
    _validate_method_definition(cls, "create_scheduler", ["self", "optimizer"])

    if cls.name in _supported_scheduler_factory_registry:
        if not force:
            raise AlreadyRegisteredError(
                f"Scheduler Factory {cls.name} is already registered. If you want ot"
                " overwrite, set force=True"
            )
        logger.warning(f"Overwriting existing registered Scheduler Factory {cls.name}.")

    _supported_scheduler_factory_registry[cls.name] = cls
    logger.info(f"Registered SchedulerFactory {cls.name}")

    return cls


def get_registered_scheduler_factory(
    name_or_class: Union[str, Type["SchedulerFactory"]],
) -> Type["SchedulerFactory"]:
    global _supported_scheduler_factory_registry

    if isinstance(name_or_class, str):
        scheduler_factory_name = name_or_class
    else:
        scheduler_factory_name = name_or_class.name

    if scheduler_factory_name not in _supported_scheduler_factory_registry:
        raise ValueError(
            f"{scheduler_factory_name} is not registered Scheduler Factory."
        )

    return _supported_scheduler_factory_registry[scheduler_factory_name]
