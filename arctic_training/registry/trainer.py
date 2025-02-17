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
from arctic_training.registry.checkpoint import register_checkpoint_engine
from arctic_training.registry.data import register_data_factory
from arctic_training.registry.model import register_model_factory
from arctic_training.registry.optimizer import register_optimizer_factory
from arctic_training.registry.scheduler import register_scheduler_factory
from arctic_training.registry.tokenizer import register_tokenizer_factory
from arctic_training.registry.utils import AlreadyRegisteredError
from arctic_training.registry.utils import _get_class_attr_type_hints
from arctic_training.registry.utils import _validate_class_attribute_set
from arctic_training.registry.utils import _validate_class_attribute_type

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer

_supported_trainer_registry: Dict[str, Type["Trainer"]] = {}


def register_trainer(cls: Type["Trainer"], force: bool = False) -> Type["Trainer"]:
    from arctic_training.checkpoint.engine import CheckpointEngine
    from arctic_training.config.trainer import TrainerConfig
    from arctic_training.data.factory import DataFactory
    from arctic_training.model.factory import ModelFactory
    from arctic_training.optimizer.factory import OptimizerFactory
    from arctic_training.scheduler.factory import SchedulerFactory
    from arctic_training.tokenizer.factory import TokenizerFactory
    from arctic_training.trainer.trainer import Trainer

    global _supported_trainer_registry

    if not issubclass(cls, Trainer):
        raise ValueError(
            f"New trainer {cls.__name__} class must be a subclass of Trainer."
        )

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_type(cls, "config", TrainerConfig)

    trainer_attributes = [
        (
            "data_factory",
            DataFactory,
            register_data_factory,
        ),
        (
            "model_factory",
            ModelFactory,
            register_model_factory,
        ),
        (
            "checkpoint_engine",
            CheckpointEngine,
            register_checkpoint_engine,
        ),
        (
            "optimizer_factory",
            OptimizerFactory,
            register_optimizer_factory,
        ),
        (
            "scheduler_factory",
            SchedulerFactory,
            register_scheduler_factory,
        ),
        (
            "tokenizer_factory",
            TokenizerFactory,
            register_tokenizer_factory,
        ),
    ]
    for attr, type_, register_class in trainer_attributes:
        _validate_class_attribute_type(cls, attr, type_)

        # Try to register the class, skip if already registered
        for attr_type_hint in _get_class_attr_type_hints(cls, attr):
            try:
                _ = register_class(attr_type_hint)
            except AlreadyRegisteredError:
                pass

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
