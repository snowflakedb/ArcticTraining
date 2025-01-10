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
from typing import Iterable
from typing import Type
from typing import Union

from arctic_training.logging import logger
from arctic_training.registry.checkpoint import get_registered_checkpoint_engine
from arctic_training.registry.checkpoint import register_checkpoint_engine
from arctic_training.registry.data import get_registered_data_factory
from arctic_training.registry.data import register_data_factory
from arctic_training.registry.model import get_registered_model_factory
from arctic_training.registry.model import register_model_factory
from arctic_training.registry.optimizer import get_registered_optimizer_factory
from arctic_training.registry.optimizer import register_optimizer_factory
from arctic_training.registry.scheduler import get_registered_scheduler_factory
from arctic_training.registry.scheduler import register_scheduler_factory
from arctic_training.registry.tokenizer import get_registered_tokenizer_factory
from arctic_training.registry.tokenizer import register_tokenizer_factory
from arctic_training.registry.utils import AlreadyRegisteredError
from arctic_training.registry.utils import _validate_class_attribute_set

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer

_supported_trainer_registry: Dict[str, Type["Trainer"]] = {}


def register_trainer(cls: Type["Trainer"], force: bool = False) -> Type["Trainer"]:
    global _supported_trainer_registry
    from arctic_training.trainer.trainer import Trainer

    if not issubclass(cls, Trainer):
        raise ValueError(
            f"New trainer {cls.__name__} class must be a subclass of Trainer."
        )

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_set(cls, "config_type")

    trainer_attributes = [
        (
            "data_factory_type",
            get_registered_data_factory,
            register_data_factory,
        ),
        (
            "model_factory_type",
            get_registered_model_factory,
            register_model_factory,
        ),
        (
            "checkpoint_engine_type",
            get_registered_checkpoint_engine,
            register_checkpoint_engine,
        ),
        (
            "optimizer_factory_type",
            get_registered_optimizer_factory,
            register_optimizer_factory,
        ),
        (
            "scheduler_factory_type",
            get_registered_scheduler_factory,
            register_scheduler_factory,
        ),
        (
            "tokenizer_factory_type",
            get_registered_tokenizer_factory,
            register_tokenizer_factory,
        ),
    ]
    for attr, get_class, register_class in trainer_attributes:
        _validate_class_attribute_set(cls, attr)

        # Coerce to list if not already
        if not isinstance(getattr(cls, attr), Iterable) or isinstance(
            getattr(cls, attr), str
        ):
            setattr(cls, attr, [getattr(cls, attr)])

        for class_type in getattr(cls, attr):
            # Try to register the class, skip if already registered
            if not issubclass(class_type, str):
                try:
                    _ = register_class(class_type)
                except AlreadyRegisteredError:
                    pass

            # Verify that the class is registered
            _ = get_class(class_type)

    if cls.name in _supported_trainer_registry:
        if not force:
            raise AlreadyRegisteredError(
                f"Trainer class {cls.name} is already registered. Use force=True to"
                " override."
            )
        logger.warning(f"Overwriting existing registered trainer {cls.name}.")

    _supported_trainer_registry[cls.name] = cls
    logger.info(f"Registered Trainer {cls.name}")

    return cls


def get_registered_trainer(
    name_or_class: Union[str, Type["Trainer"]],
) -> Type["Trainer"]:
    global _supported_trainer_registry

    if isinstance(name_or_class, str):
        trainer_name = name_or_class
    else:
        trainer_name = name_or_class.name

    if trainer_name not in _supported_trainer_registry:
        raise ValueError(f"{trainer_name} is not a registered Trainer.")

    return _supported_trainer_registry[trainer_name]
