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
from abc import ABC
from abc import ABCMeta
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import get_args
from typing import get_origin
from typing import get_type_hints

from arctic_training.logging import logger

if TYPE_CHECKING:
    from arctic_training.checkpoint.engine import CheckpointEngine
    from arctic_training.data.factory import DataFactory
    from arctic_training.data.source import DataSource
    from arctic_training.model.factory import ModelFactory
    from arctic_training.optimizer.factory import OptimizerFactory
    from arctic_training.scheduler.factory import SchedulerFactory
    from arctic_training.tokenizer.factory import TokenizerFactory
    from arctic_training.trainer.trainer import Trainer

TRegisteredClass = TypeVar("TRegisteredClass")


def register(cls: Optional[Type] = None, force: bool = False) -> Union[Callable, Type]:
    logger.warning(
        "The `@register` decorator is deprecated and will be removed in a future"
        " release. ArcticTraining base classes now use"
        " `arctic_training.registry.RegistryMeta` metaclass for registration. This"
        " means that custom classes are automatically registered during declaration and"
        " explicit registration via the decorator is not necessary."
    )

    # If called without parentheses, cls will be the class itself
    if cls and isinstance(cls, type):
        return cls

    # Otherwise, return a decorator that takes cls later
    def decorator(cls):
        return cls

    return decorator


class RegistryMeta(ABCMeta):
    """A flexible metaclass that registers subclasses and enforces validation."""

    # {BaseClassName: {SubClassName: SubClassType}}
    _registry: Dict[str, Dict[str, Type]] = {}

    def __new__(
        mcs: Type["RegistryMeta"], name: str, bases: Tuple, class_dict: Dict
    ) -> Type:
        """Creates a new class, registers it, and ensures it has `_validate_subclass`."""
        cls: Type = super().__new__(mcs, name, bases, class_dict)

        # Don't register the base classes themselves
        if any(base for base in bases if isinstance(base, ABCMeta) and base is not ABC):
            # Assuming single inheritance for base class
            base_type: str = bases[0].__name__

            # Ensure the subclass defines `_validate_subclass`
            if not hasattr(cls, "_validate_subclass") or not callable(
                getattr(cls, "_validate_subclass")
            ):
                raise TypeError(
                    f"Class {cls.__name__} must define a `_validate_subclass` method."
                )

            cls._validate_subclass()

            # We know that class has "name" defined if it passes `_validate_subclass`
            registry_name = class_dict["name"]

            # Register subclass
            if base_type not in mcs._registry:
                mcs._registry[base_type] = {}
            mcs._registry[base_type][registry_name] = cls

        return cls


def get_registered_class(class_type: str, name: str) -> Type:
    if name not in RegistryMeta._registry.get(class_type, {}):
        raise ValueError(f"{name} is not a registered {class_type}.")
    return RegistryMeta._registry[class_type][name]


def get_registered_checkpoint_engine(name: str) -> Type["CheckpointEngine"]:
    return get_registered_class(class_type="CheckpointEngine", name=name)


def get_registered_data_factory(name: str) -> Type["DataFactory"]:
    return get_registered_class(class_type="DataFactory", name=name)


def get_registered_data_source(name: str) -> Type["DataSource"]:
    return get_registered_class(class_type="DataSource", name=name)


def get_registered_model_factory(name: str) -> Type["ModelFactory"]:
    return get_registered_class(class_type="ModelFactory", name=name)


def get_registered_optimizer_factory(name: str) -> Type["OptimizerFactory"]:
    return get_registered_class(class_type="OptimizerFactory", name=name)


def get_registered_scheduler_factory(name: str) -> Type["SchedulerFactory"]:
    return get_registered_class(class_type="SchedulerFactory", name=name)


def get_registered_tokenizer_factory(name: str) -> Type["TokenizerFactory"]:
    return get_registered_class(class_type="TokenizerFactory", name=name)


def get_registered_trainer(name: str) -> Type["Trainer"]:
    return get_registered_class(class_type="Trainer", name=name)


def _validate_method_definition(
    cls: Type, method_name: str, method_params: List[str] = []
) -> None:
    method = cls.__dict__[method_name]
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


def _validate_class_attribute_set(cls: Type, attribute: str) -> None:
    if not getattr(cls, attribute, None):
        raise ValueError(f"{cls.__name__} must define {attribute} attribute.")


def _validate_class_attribute_type(cls: Type, attribute: str, type_: Type) -> None:
    for attr_type_hint in _get_class_attr_type_hints(cls, attribute):
        if not issubclass(attr_type_hint, type_):
            raise TypeError(
                f"{cls.__name__}.{attribute} must be an instance of {type_.__name__}."
                f" But got {attr_type_hint.__name__}."
            )


def _get_class_attr_type_hints(cls: Type, attribute: str) -> Tuple[Type]:
    cls_type_hints = get_type_hints(cls)
    if get_origin(cls_type_hints[attribute]) is Union:
        attribute_type_hints = get_args(cls_type_hints[attribute])
    else:
        attribute_type_hints = (cls_type_hints[attribute],)
    return attribute_type_hints
