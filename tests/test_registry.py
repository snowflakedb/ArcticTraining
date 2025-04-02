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

from abc import ABC
from abc import abstractmethod
from typing import Union

import pytest
from loguru import logger

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.exceptions import RegistryError
from arctic_training.exceptions import RegistryValidationError
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method
from arctic_training.registry import get_registered_class
from arctic_training.registry import register


@pytest.fixture(scope="function", autouse=True)
def reset_registry():
    """Fixture to reset the registry before each test."""
    original_registry = RegistryMeta._registry
    RegistryMeta._registry = {}
    yield
    RegistryMeta._registry = original_registry


@pytest.fixture
def loguru_caplog():
    """Fixture to capture Loguru logs in pytest."""
    import logging

    class PropagateHandler(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)

    logger.remove()  # Remove existing handlers to prevent duplicate logs
    logger.add(PropagateHandler(), format="{message}")  # Add handler for pytest to capture

    yield

    logger.remove()  # Clean up after test


class MyStr(str):
    """Subclass of str for testing type validation"""

    pass


class ValidationTestClass:
    """Class with various attributes and methods for testing validation"""

    set_attr: str = "test"
    unset_attr: str

    type_attr: str
    subclass_type_attr: MyStr
    union_type_attr: Union[str, MyStr]
    no_type_attr = None

    @staticmethod
    def method_without_args():
        pass

    def method_with_args(self, arg1, arg2):
        pass


def test_validate_class_method_no_args():
    _validate_class_method(ValidationTestClass, "method_without_args")


def test_validate_class_method_with_args():
    _validate_class_method(ValidationTestClass, "method_with_args", ["self", "arg1", "arg2"])


def test_validate_class_method_fail():
    with pytest.raises(RegistryValidationError):
        # Incorrect arguments for method_with_args
        _validate_class_method(ValidationTestClass, "method_with_args", ["self", "arg1"])


def test_validate_class_attribute_set():
    _validate_class_attribute_set(ValidationTestClass, "set_attr")


def test_validate_class_attribute_set_fail():
    with pytest.raises(RegistryValidationError):
        # Attribute is unset
        _validate_class_attribute_set(ValidationTestClass, "unset_attr")


def test_validate_class_attribute_type():
    _validate_class_attribute_type(ValidationTestClass, "type_attr", str)


def test_validate_class_attribute_type_subclass():
    # subclass_type_attr has type MyStr, which is a subclass of str
    _validate_class_attribute_type(ValidationTestClass, "subclass_type_attr", str)


def test_validate_class_attribute_type_union():
    # union_type_attr has type Union[str, MyStr], which are both a subclass of str
    _validate_class_attribute_type(ValidationTestClass, "union_type_attr", str)


def test_validate_class_attribute_type_fail():
    with pytest.raises(RegistryValidationError):
        # No type hinting for attribute
        _validate_class_attribute_type(ValidationTestClass, "no_type_attr", str)


def test_validate_class_attribute_type_union_fail():
    with pytest.raises(RegistryValidationError):
        # str is not a subclass of MyStr
        _validate_class_attribute_type(ValidationTestClass, "union_attr", MyStr)


def test_registration_decorator_deprecation(loguru_caplog, caplog):
    class BaseClass(ABC, CallbackMixin, metaclass=RegistryMeta):
        @classmethod
        def _validate_subclass(cls):
            pass

    with caplog.at_level("WARNING"):

        @register
        class RegisteredClass(BaseClass):
            name = "test_class"

    assert "The @register decorator is deprecated" in caplog.text, f"Wrong warning message: {caplog.text}"
    assert RegisteredClass.name in RegistryMeta._registry[BaseClass.__name__], "Class not registered correctly"


def test_registration():
    class BaseClass(ABC, metaclass=RegistryMeta):
        name: str

        @classmethod
        def _validate_subclass(cls):
            pass

        @abstractmethod
        def method(self):
            raise NotImplementedError

    class RegisteredClass(BaseClass):
        name = "test_class"

        def method(self):
            pass

    assert RegisteredClass.name in RegistryMeta._registry[BaseClass.__name__], "Class not registered correctly"
    assert RegisteredClass is RegistryMeta._registry[BaseClass.__name__][RegisteredClass.name], "Class does not match"


def test_registration_multi_inheritance():
    class BaseClass(ABC, CallbackMixin, metaclass=RegistryMeta):
        @classmethod
        def _validate_subclass(cls):
            _validate_class_attribute_set(cls, "name")
            _validate_class_method(cls, "method1", ["self"])
            _validate_class_method(cls, "method2", ["self", "arg1"])

        @abstractmethod
        def method1(self):
            raise NotImplementedError

        @abstractmethod
        def method2(self, arg1):
            raise NotImplementedError

    class RegisteredSubClass(BaseClass):
        name = "sub_class"

        def method1(self):
            pass

        def method2(self, arg1):
            pass

    class RegisteredSubSubClass(RegisteredSubClass):
        name = "sub_sub_class"

        def method2(self, arg1):
            pass

    assert RegisteredSubClass.name in RegistryMeta._registry[BaseClass.__name__], "Class not registered correctly"
    assert RegisteredSubSubClass.name in RegistryMeta._registry[BaseClass.__name__], "Class not registered correctly"


def test_registration_fail():
    class BaseClass(ABC, CallbackMixin, metaclass=RegistryMeta):
        @classmethod
        def _validate_subclass(cls):
            pass

    # Populate registry with something
    class RegisteredClass(BaseClass):
        name = "test_class"

    with pytest.raises(RegistryError):
        # No classes of type DoesNotExist have been registered
        get_registered_class(class_type="DoesNotExist", name="test_class")

    with pytest.raises(RegistryError):
        # Classes of type BaseClass have been registered, but does_not_exist has not
        get_registered_class(class_type="BaseClass", name="does_not_exist")


def test_registration_validation_missing_fail():
    with pytest.raises(RegistryValidationError):

        class BaseClass(ABC, CallbackMixin, metaclass=RegistryMeta):
            # No _validate_subclass method defined
            pass


def test_registration_validation_non_callable_fail():
    with pytest.raises(RegistryValidationError):

        class BaseClass(ABC, CallbackMixin, metaclass=RegistryMeta):
            _validate_subclass = "fail"  # Not a callable


def test_registration_validation_bad_args_fail():
    with pytest.raises(RegistryValidationError):

        class BaseClass(ABC, CallbackMixin, metaclass=RegistryMeta):
            @classmethod
            def _validate_subclass(cls, other_arg):  # Too many args
                pass


def test_registration_validation_missing_args_fail():
    with pytest.raises(RegistryValidationError):

        class BaseClass(ABC, CallbackMixin, metaclass=RegistryMeta):
            @classmethod
            def _validate_subclass():  # Not enough args
                pass


def test_registration_validation_no_name_fail():
    class BaseClass(ABC, CallbackMixin, metaclass=RegistryMeta):
        name: str

        @classmethod
        def _validate_subclass(cls):
            pass

    with pytest.raises(RegistryValidationError):

        class RegisteredClass(BaseClass):
            # name is not defined
            pass


def test_registration_validation_literal_fail():
    class BaseClass(ABC, CallbackMixin, metaclass=RegistryMeta):
        name: str

        @classmethod
        def _validate_subclass(cls):
            raise RegistryValidationError  # Always fail

    with pytest.raises(RegistryValidationError):

        class RegisteredClass(BaseClass):
            name = "test_class"


def test_registration_already_registered_fail():
    class BaseClass(ABC, CallbackMixin, metaclass=RegistryMeta):
        name: str

        @classmethod
        def _validate_subclass(cls):
            pass

    class RegisteredClass(BaseClass):
        name = "test_class"

    with pytest.raises(RegistryValidationError):

        class NewRegisteredClass(BaseClass):
            name = "test_class"
