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
from typing import Union
from typing import get_args
from typing import get_origin
from typing import get_type_hints

from arctic_training.exceptions import RegistryError
from arctic_training.exceptions import RegistryValidationError
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


def register(cls: Optional[Type] = None, force: bool = False) -> Union[Callable, Type]:
    logger.warning(
        "The @register decorator is deprecated and will be removed in a future"
        " release. ArcticTraining base classes now use"
        " arctic_training.registry.RegistryMeta metaclass for registration. This"
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
    """A metaclass that registers subclasses of ArcticTraining base classes."""

    # {BaseClassName: {SubClassName: SubClassType}}
    _registry: Dict[str, Dict[str, Type]] = {}

    def __new__(mcs: Type["RegistryMeta"], name: str, bases: Tuple, class_dict: Dict) -> Type:
        """Creates a new class, validates it, and registers it."""
        cls: Type = super().__new__(mcs, name, bases, class_dict)

        _validate_class_method(cls, "_validate_subclass", ["cls"])

        # Don't register the base classes themselves
        if mcs._is_base_class(bases):
            return cls

        # Validate the subclass
        _validate_class_attribute_set(cls, "name")
        cls._validate_subclass()

        # Iterate up the inheritance chain to find the base class
        while not mcs._is_base_class(bases):
            root_base = bases[0]  # Assuming single inheritance for sub classes
            bases = bases[0].__bases__

        # Register subclass
        base_type: str = root_base.__name__
        registry_name = class_dict["name"]
        if base_type not in mcs._registry:
            mcs._registry[base_type] = {}
        if registry_name in mcs._registry[base_type]:
            raise RegistryValidationError(f"{registry_name} is already registered as a {base_type}.")
        mcs._registry[base_type][registry_name] = cls

        return cls

    @staticmethod
    def _is_base_class(bases: Tuple[Type]) -> bool:
        return any(base is ABC for base in bases)


def get_registered_class(class_type: str, name: str) -> Type:
    if class_type not in RegistryMeta._registry:
        raise RegistryError(
            f"No classes of type {class_type} have been registered. Ensure that"
            f" {class_type} is a base class that uses the RegistryMeta metaclass (e.g.,"
            " class MyBaseClass(ABC,"
            " metaclass=arctic_training.registry.RegistryMeta)). The base class will"
            " not be registered, but any inheriting subclasses will be registered"
            " automatically. Ensure that the intended registered class has been"
            " imported before calling a get_registered_*() function."
        )
    if name not in RegistryMeta._registry[class_type]:
        raise RegistryError(
            f"{name} is not a registered {class_type}. Ensure that {name} is a subclass"
            f" of {class_type} and that the class has been imported before calling a"
            " get_registered_*() function. Available registered classes of type"
            f" {class_type} are: {list(RegistryMeta._registry[class_type].keys())}"
        )
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


def _validate_class_method(cls: Type, method_name: str, expected_args: List[str] = []) -> None:
    if not hasattr(cls, method_name):
        raise RegistryValidationError(f"{cls.__name__} must define a '{method_name}' method.")

    method = getattr(cls, method_name)
    if not callable(method):
        raise RegistryValidationError(f"{cls.__name__}.{method_name} must be a callable method.")

    if inspect.ismethod(method):
        method = method.__func__  # Unwrap class method

    sig = inspect.signature(method)
    actual_args = set(sig.parameters.keys())
    if actual_args != set(expected_args):
        raise RegistryValidationError(
            f"{cls.__name__}.{method_name} must accept exactly"
            f" {set(expected_args)} as parameters, but got {actual_args}."
        )


def _validate_class_attribute_set(cls: Type, attribute: str) -> None:
    if not hasattr(cls, attribute):
        raise RegistryValidationError(f"{cls.__name__} must define a '{attribute}' attribute.")


def _validate_class_attribute_type(cls: Type, attribute: str, type_: Type) -> None:
    class_attr_type_hints = _get_class_attr_type_hints(cls, attribute)
    if len(class_attr_type_hints) == 0:
        raise RegistryValidationError(f"{cls.__name__}.{attribute} must have a type hint.")

    bad_types = []
    for attr_type_hint in class_attr_type_hints:
        if not issubclass(attr_type_hint, type_):
            bad_types.append(attr_type_hint)

    if len(bad_types) != 0:
        raise RegistryValidationError(
            f"{cls.__name__}.{attribute} must define one or more types that are a"
            f" subclass of {type_.__name__} for the {attribute} attribute, but we found"
            " the following types are not subclasses:"
            f" {[t.__name__ for t in bad_types]}."
        )


def _get_class_attr_type_hints(cls: Type, attribute: str) -> Union[Tuple[()], Tuple[Type, ...]]:
    cls_type_hints = get_type_hints(cls)
    if attribute not in cls_type_hints:
        return tuple()
    elif get_origin(cls_type_hints[attribute]) is Union:
        attribute_type_hints = get_args(cls_type_hints[attribute])
    else:
        attribute_type_hints = (cls_type_hints[attribute],)
    return attribute_type_hints
