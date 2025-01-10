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

import inspect
from typing import TYPE_CHECKING
from typing import List
from typing import Type
from typing import Union

if TYPE_CHECKING:
    from arctic_training.checkpoint.engine import CheckpointEngine
    from arctic_training.config.base import BaseConfig
    from arctic_training.data.factory import DataFactory
    from arctic_training.data.source import DataSource
    from arctic_training.model.factory import ModelFactory
    from arctic_training.optimizer.factory import OptimizerFactory
    from arctic_training.scheduler.factory import SchedulerFactory
    from arctic_training.tokenizer.factory import TokenizerFactory
    from arctic_training.trainer.trainer import Trainer

RegistryClassTypes = Union[
    Type["CheckpointEngine"],
    Type["BaseConfig"],
    Type["DataFactory"],
    Type["DataSource"],
    Type["ModelFactory"],
    Type["OptimizerFactory"],
    Type["SchedulerFactory"],
    Type["TokenizerFactory"],
    Type["Trainer"],
]


class AlreadyRegisteredError(Exception):
    pass


def _validate_method_definition(
    cls: RegistryClassTypes, method_name: str, method_params: List[str] = []
) -> None:
    # Avoid circular import
    from arctic_training.checkpoint.engine import CheckpointEngine
    from arctic_training.config.base import BaseConfig
    from arctic_training.data.factory import DataFactory
    from arctic_training.data.source import DataSource
    from arctic_training.model.factory import ModelFactory
    from arctic_training.optimizer.factory import OptimizerFactory
    from arctic_training.scheduler.factory import SchedulerFactory
    from arctic_training.tokenizer.factory import TokenizerFactory
    from arctic_training.trainer.trainer import Trainer

    BaseClasses = [
        DataFactory,
        DataSource,
        Trainer,
        BaseConfig,
        CheckpointEngine,
        ModelFactory,
        OptimizerFactory,
        SchedulerFactory,
        TokenizerFactory,
    ]

    for subclass in cls.__mro__:
        # Skip the base class itself
        if subclass in BaseClasses:
            break
        if method_name not in subclass.__dict__:
            continue
        method = subclass.__dict__[method_name]
        if not callable(method):
            raise ValueError(f"{cls.__name__}.{method_name} must be a callable method.")
        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        params_names = set(p.name for p in params)
        if not params_names == set(method_params):
            raise ValueError(
                f"{cls.__name__}.{method_name} must accept exactly"
                f" {set(method_params)} as parameters, but got {params_names}."
            )
    else:
        raise ValueError(
            f"{cls.__name__} must implement its own '{method_name}' method."
        )


def _validate_class_attribute_set(cls: RegistryClassTypes, attribute: str) -> None:
    if not getattr(cls, attribute, None):
        raise ValueError(f"{cls.__name__} must define {attribute} attribute.")
