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
from typing import Any
from typing import Callable
from typing import Dict
from typing import Tuple
from typing import Union

from arctic_training.logging import logger


class Callback:
    def __init__(self, event: str, fn: Callable, method: Callable) -> None:
        logger.debug(f"Initializing Callback for {method} with event={event} and fn={fn.__name__}")
        self.event = event
        self.fn = fn
        self.method = method
        self.cb_fn_sig = inspect.signature(fn)
        self.method_sig = inspect.signature(method)
        self.params_to_pass: Tuple = tuple()
        self.pass_return_val: bool = True

        self._validate_fn_sig()

    @property
    def is_pre_cb(self) -> bool:
        return "pre" in self.event

    @property
    def is_post_cb(self) -> bool:
        return "post" in self.event

    def _validate_fn_sig(self) -> None:
        if self.is_pre_cb:
            self._validate_pre_fn_sig()
        elif self.is_post_cb:
            self._validate_post_fn_sig()
        else:
            raise ValueError(f"Invalid event type: {self.event}. Expected 'pre' or 'post' in event name.")

    def _validate_pre_fn_sig(self) -> None:
        callback_params = list(self.cb_fn_sig.parameters.values())
        method_params = list(self.method_sig.parameters.keys())
        params_to_pass = []
        for param in callback_params:
            if param.default != param.empty:
                raise ValueError(
                    f"Callback function {self.fn.__name__} has a default value for"
                    f" parameter {param.name}. Callbacks should not have default"
                    " values."
                )
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise ValueError(
                    f"Callback function {self.fn.__name__} has a *args or **kwargs"
                    " parameter. Callbacks should not have *args or **kwargs."
                )
            if param.name not in method_params:
                raise ValueError(
                    f"Callback function {self.fn.__name__} has a parameter"
                    f" {param.name} that is not present in the method signature."
                )
            params_to_pass.append(param.name)

        if len(params_to_pass) < 1:
            raise ValueError(f"Pre-callback function {self.fn} should have at least one parameter: `self`.")

        self.params_to_pass = tuple(params_to_pass[1:])

    def _validate_post_fn_sig(self) -> None:
        if self.method_sig.return_annotation == inspect.Signature.empty:
            logger.warning(
                f"Callback-wrapped method {self.method} does not have a return"
                f" signature. Validation of post-callback function {self.fn} will be"
                " skipped."
            )
        elif self.method_sig.return_annotation is None:
            self.pass_return_val = False
            if len(self.cb_fn_sig.parameters) != 1:
                raise ValueError(f"Post-callback function {self.fn} should have at most one parameter: `self`.")
        elif len(self.cb_fn_sig.parameters) != 2:
            raise ValueError(
                f"Post-callback function {self.fn} should have exactly two parameters:"
                " `self` and the return value of the method it wraps."
            )

    def _run_pre_callback(self, obj: object, args: Tuple[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # Combine args and kwargs to match method signature and get defaults
        bound_args = self.method_sig.bind(obj, *args, **kwargs)
        bound_args.apply_defaults()
        method_kwargs = {p: v for p, v in bound_args.arguments.items() if p not in ("self", "args", "kwargs")}

        params_to_pass = {param: method_kwargs[param] for param in self.params_to_pass}
        returned_args = self.fn(obj, **params_to_pass)

        if len(params_to_pass) == 0:
            if returned_args is not None:
                raise ValueError(f"Pre-callback function {self.fn} should not return any values.")
            else:
                returned_args = tuple()
        elif len(params_to_pass) == 1:
            returned_args = (returned_args,)
        elif len(returned_args) != len(params_to_pass):
            raise ValueError(
                f"Pre-callback function {self.fn} returned"
                f" {len(returned_args)} arguments, but expected"
                f" {len(params_to_pass)} arguments."
            )

        returned_kwargs = {param: returned_args[i] for i, param in enumerate(self.params_to_pass)}
        method_kwargs.update(returned_kwargs)
        return method_kwargs

    def _run_post_callback(self, obj: object, return_val: Any) -> Union[Any, Tuple[Any]]:
        if self.pass_return_val:
            try:
                return self.fn(obj, return_val)
            except Exception as e:
                logger.error(
                    f"Error running post-callback {self.fn} for event {self.event}."
                    " This may be due to the callback not accepting a return value in"
                    " its input signature."
                )
                raise e
        else:
            return self.fn(obj)

    def __call__(self, obj: object, args: Union[Any, Tuple[Any]], kwargs: Dict[str, Any]) -> Tuple[Tuple, Dict]:
        logger.debug(
            f"Running callback {self.fn} in object {obj} for event {self.event} with arg(s) {args} and kwargs {kwargs}"
        )
        if self.is_pre_cb:
            return tuple(), self._run_pre_callback(obj, args, kwargs)
        elif self.is_post_cb:
            return self._run_post_callback(obj, args), dict()
        else:
            raise ValueError(f"Invalid event type: {self.event}. Expected 'pre' or 'post' in event name.")

    def __repr__(self) -> str:
        return f"<Callback event={self.event} fn={self.fn.__name__}>"
