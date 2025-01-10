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
    from arctic_training.checkpoint.engine import CheckpointEngine

from arctic_training.logging import logger
from arctic_training.registry.utils import AlreadyRegisteredError
from arctic_training.registry.utils import _validate_class_attribute_set

_supported_checkpoint_registry: Dict[str, Type["CheckpointEngine"]] = {}


def register_checkpoint_engine(
    cls: Type["CheckpointEngine"], force: bool = False
) -> Type["CheckpointEngine"]:
    global _supported_checkpoint_registry
    from arctic_training.checkpoint.engine import CheckpointEngine

    if not issubclass(cls, CheckpointEngine):
        raise ValueError(
            f"New checkpoint engine {cls.__name__} class must be a subclass of"
            " CheckpointEngine."
        )

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_set(cls, "config_type")

    if cls.name in _supported_checkpoint_registry:
        if not force:
            raise AlreadyRegisteredError(
                f"Checkpoint engine {cls.name} is already registered. If you want to"
                " overwrite, set force=True."
            )
        logger.warning(f"Overwriting existing registered checkpoint engine {cls.name}.")

    _supported_checkpoint_registry[cls.name] = cls
    logger.info(f"Registered CheckpointEngine {cls.name}")

    return cls


def get_registered_checkpoint_engine(
    name_or_class: Union[str, Type["CheckpointEngine"]],
) -> Type["CheckpointEngine"]:
    global _supported_checkpoint_registry

    if isinstance(name_or_class, str):
        checkpoint_engine_name = name_or_class
    else:
        checkpoint_engine_name = name_or_class.name

    if checkpoint_engine_name not in _supported_checkpoint_registry:
        raise ValueError(
            f"{checkpoint_engine_name} checkpoint engine is not registered!"
        )

    return _supported_checkpoint_registry[checkpoint_engine_name]
