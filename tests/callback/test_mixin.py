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
from abc import abstractmethod

import pytest

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper


class BaseClass(ABC, CallbackMixin):
    val = 16

    @callback_wrapper("no-args")
    def no_args(self) -> int:
        return self.val * 42

    @callback_wrapper("one-arg")
    def one_arg(self, val: int) -> int:
        return val * 10

    @callback_wrapper("multi-args")
    def multi_args(self, val1: int, val2: int) -> int:
        return val1 * val2

    @callback_wrapper("no-out")
    def no_out(self, val: int) -> None:
        self.val = val + 42

    @callback_wrapper("any-out")
    def any_out(self, val: int) -> int:
        return val * 2

    @abstractmethod
    @callback_wrapper("abstract")
    def abstract(self, val: int) -> int:
        pass


class SubClass(BaseClass):
    val = 32

    def abstract(self, val: int) -> int:
        return val**2


class SubSubClass(SubClass):
    val = 64

    def abstract(self, val: int) -> int:
        return val**3


def callback_no_args_fn(self) -> None:
    self.val = self.val + 123


def callback_one_arg_fn(self, val: int) -> int:
    return val * 12


def callback_multi_args_fn(self, val1: int, val2: int) -> int:
    return val1 * 3, val2 * 5


@pytest.fixture
def cls(base_cls, method_name, pre_callback, post_callback):
    class ClassWithCallbacks(base_cls):
        callbacks = [
            ("pre-" + method_name.replace("_", "-"), pre_callback),
            ("post-" + method_name.replace("_", "-"), post_callback),
        ]

    return ClassWithCallbacks


@pytest.fixture
def input_args(base_cls, num_args):
    if num_args == 0:
        return ()
    if num_args == 1:
        return (base_cls.val,)
    if num_args == 2:
        return (base_cls.val, base_cls.val)
    raise ValueError("Invalid number of arguments")


@pytest.fixture
def expected_val(base_cls, method_name, num_args, input_args, pre_callback, post_callback):
    obj = base_cls()

    new_args = pre_callback(obj, *input_args)
    if num_args == 0:
        new_args = tuple()
    elif num_args == 1:
        new_args = (new_args,)

    return_val = getattr(obj, method_name)(*new_args)

    if return_val is None:
        post_callback(obj)
        return obj.val
    else:
        return post_callback(obj, return_val)


@pytest.fixture
def num_args(base_cls, method_name):
    method = getattr(base_cls, method_name)
    sig = inspect.signature(method)
    return len(sig.parameters) - 1  # account for self


@pytest.fixture
def pre_callback(num_args):
    if num_args == 0:
        return callback_no_args_fn
    if num_args == 1:
        return callback_one_arg_fn
    if num_args == 2:
        return callback_multi_args_fn
    raise ValueError("Invalid number of arguments")


@pytest.fixture
def post_callback(base_cls, method_name):
    method = getattr(base_cls, method_name)
    sig = inspect.signature(method)
    if sig.return_annotation is None:
        return callback_no_args_fn
    return callback_one_arg_fn


@pytest.mark.parametrize("base_cls", [SubClass, SubSubClass], ids=["SubClass", "SubSubClass"])
@pytest.mark.parametrize("method_name", ["no_args", "one_arg", "multi_args", "no_out", "any_out"])
def test_callback_inputs_outputs(cls, method_name, input_args, expected_val):
    obj = cls()
    val = getattr(obj, method_name)(*input_args)
    if val is None:
        val = obj.val
    assert val == expected_val
