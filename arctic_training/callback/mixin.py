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

import functools
import inspect
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type

from arctic_training.callback.callback import Callback
from arctic_training.callback.logging import post_init_callback_ordering_cb
from arctic_training.callback.logging import post_init_log_cb
from arctic_training.callback.logging import pre_init_log_cb
from arctic_training.logging import logger

WRAPPER_NAME_ATTR = "__at_wrapped_name__"


def callback_wrapper(name: str):
    """A decorator to wrap a method with pre- and post-callbacks."""

    def decorator(method):
        if hasattr(method, WRAPPER_NAME_ATTR):
            return method

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            args, kwargs = self._run_callbacks(f"pre-{name}", args, kwargs)
            return_val = method(self, *args, **kwargs)
            return_val, _ = self._run_callbacks(f"post-{name}", return_val, {})
            return return_val

        setattr(wrapper, WRAPPER_NAME_ATTR, name)
        return wrapper

    return decorator


class CallbackMixin:
    """A mixin class that provides callback functionality to a class."""

    _class_callbacks: List[Tuple[str, Callable]] = []

    _initialized_callbacks: List[Callback] = []

    callbacks: List[Tuple[str, Callable]] = [
        pre_init_log_cb,
        post_init_log_cb,
        post_init_callback_ordering_cb,
    ]
    """ A list of callbacks that are applied to the class. """

    @callback_wrapper("init")
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __new__(cls: Type["CallbackMixin"], *args, **kwargs) -> "CallbackMixin":
        # Gather all callbacks from cls._class_callbacks and class methods.
        cls._class_callbacks = cls._get_all_callbacks()

        # Initialize callbacks
        cls._initialized_callbacks = cls._init_callbacks()

        instance = super().__new__(cls)
        return instance

    def __init_subclass__(cls: Type["CallbackMixin"]) -> None:
        super().__init_subclass__()

        # If a sub-class redefines a method that was previously wrapped in a
        # parent class, we need to re-wrap it.
        cls._rewrap_class_methods()

        # Accumulate callbacks from parent classes
        cls._class_callbacks = cls._class_callbacks.copy()
        cls._class_callbacks.extend([cb for cb in cls.callbacks if cb not in cls._class_callbacks])

    @classmethod
    def _rewrap_class_methods(cls: Type["CallbackMixin"]) -> None:
        # Wrap any methods that were previously wrapped in the parent classes.
        # This is necessary to keep callbacks working even when base trainer
        # classes like loss and step are overridden.
        for parent_class in cls.__bases__:
            for name, member in inspect.getmembers(parent_class, predicate=inspect.isfunction):
                if hasattr(member, WRAPPER_NAME_ATTR):
                    wrapper_name = getattr(member, WRAPPER_NAME_ATTR)
                    logger.debug(f"Rewrapping method {cls.__name__}.{name}")
                    original_method = getattr(cls, name)
                    wrapped_method = callback_wrapper(wrapper_name)(original_method)
                    setattr(cls, name, wrapped_method)

    @classmethod
    def _get_all_callback_event_methods(
        cls: Type["CallbackMixin"],
    ) -> Dict[str, Callable]:
        wrapped_methods: Dict[str, Callable] = {}
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
            if hasattr(member, "__at_wrapped_name__"):
                wrapped_name = getattr(member, WRAPPER_NAME_ATTR)
                wrapped_methods["pre-" + wrapped_name] = member
                wrapped_methods["post-" + wrapped_name] = member
        return wrapped_methods

    @classmethod
    def _get_all_callbacks(
        cls: Type["CallbackMixin"],
    ) -> List[Tuple[str, Callable]]:
        callbacks = cls._class_callbacks
        callback_re = re.compile(r"(pre|post)_(.+)_callback")
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
            match = callback_re.match(name)
            if match:
                event = f"{match.group(1)}-{match.group(2).replace('_', '-')}"
                if (event, member) not in callbacks:
                    callbacks.append((event, member))

        return callbacks

    @classmethod
    def _init_callbacks(cls: Type["CallbackMixin"]) -> List[Callback]:
        wrapped_methods = cls._get_all_callback_event_methods()

        initialized_callbacks = []
        for event, fn in cls._class_callbacks:
            if event not in wrapped_methods:
                raise ValueError(
                    f"Callback {fn} for event {event} does not have a corresponding callback-wrapped method."
                )

            method = wrapped_methods[event]

            cb = Callback(event, fn, method)
            initialized_callbacks.append(cb)

        return initialized_callbacks

    def _run_callbacks(
        self,
        name: str,
        args: Tuple[Any],
        kwargs: Dict[str, Any],
    ) -> Tuple[Tuple, Dict]:
        for callback in self._initialized_callbacks:
            if callback.event == name:
                args, kwargs = callback(self, args, kwargs)
        return args, kwargs
