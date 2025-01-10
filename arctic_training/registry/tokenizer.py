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

if TYPE_CHECKING:
    from arctic_training.tokenizer.factory import TokenizerFactory

from arctic_training.registry.utils import AlreadyRegisteredError
from arctic_training.registry.utils import _validate_class_attribute_set
from arctic_training.registry.utils import _validate_method_definition

_supported_tokenizer_factory_registry: Dict[str, Type["TokenizerFactory"]] = {}


def register_tokenizer_factory(
    cls: Type["TokenizerFactory"], force: bool = False
) -> Type["TokenizerFactory"]:
    global _supported_tokenizer_factory_registry
    from arctic_training.tokenizer.factory import TokenizerFactory

    if not issubclass(cls, TokenizerFactory):
        raise ValueError(
            f"New Tokenizer Factory {cls.__name__} clss must be a subclass of"
            " TokenizerFactory."
        )

    _validate_class_attribute_set(cls, "name")
    _validate_class_attribute_set(cls, "config_type")
    _validate_method_definition(cls, "create_tokenizer", ["self"])

    if cls.name in _supported_tokenizer_factory_registry:
        if not force:
            raise AlreadyRegisteredError(
                f"Model Factory {cls.name} is already registered. If you want ot"
                " overwrite, set force=True"
            )
        logger.warning(f"Overwriting existing registered Tokenizer Factory {cls.name}.")

    _supported_tokenizer_factory_registry[cls.name] = cls
    logger.info(f"Registered TokenizerFactory {cls.name}")

    return cls


def get_registered_tokenizer_factory(
    name_or_class: Union[str, Type["TokenizerFactory"]],
) -> Type["TokenizerFactory"]:
    global _supported_tokenizer_factory_registry

    if isinstance(name_or_class, str):
        tokenizer_factory_name = name_or_class
    else:
        tokenizer_factory_name = name_or_class.name

    if tokenizer_factory_name not in _supported_tokenizer_factory_registry:
        raise ValueError(
            f"{tokenizer_factory_name} is not a registered Tokenizer Factory!"
        )

    return _supported_tokenizer_factory_registry[tokenizer_factory_name]
