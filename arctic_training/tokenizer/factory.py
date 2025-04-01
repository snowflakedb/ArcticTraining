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
from typing import TYPE_CHECKING
from typing import Optional

from transformers import PreTrainedTokenizer

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.tokenizer import TokenizerConfig
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


class TokenizerFactory(ABC, CallbackMixin, metaclass=RegistryMeta):
    """Base class for all tokenizer factories."""

    name: str
    """
    The name of the tokenizer factory. This is used to identify the tokenizer
    factory in the registry.
    """

    config: TokenizerConfig
    """
    The configuration class for the tokenizer factory. This is used to validate
    the configuration passed to the factory.
    """

    @classmethod
    def _validate_subclass(cls) -> None:
        _validate_class_attribute_set(cls, "name")
        _validate_class_attribute_type(cls, "config", TokenizerConfig)
        _validate_class_method(cls, "create_tokenizer", ["self"])

    def __init__(self, trainer: "Trainer", tokenizer_config: Optional[TokenizerConfig] = None) -> None:
        if tokenizer_config is None:
            tokenizer_config = trainer.config.tokenizer

        self._trainer = trainer
        self.config = tokenizer_config

    def __call__(self) -> PreTrainedTokenizer:
        tokenizer = self.create_tokenizer()
        return tokenizer

    @property
    def trainer(self) -> "Trainer":
        return self._trainer

    @property
    def device(self) -> str:
        return self.trainer.device

    @property
    def world_size(self) -> int:
        return self.trainer.world_size

    @property
    def global_rank(self) -> int:
        return self.trainer.global_rank

    @abstractmethod
    @callback_wrapper("create-tokenizer")
    def create_tokenizer(self) -> PreTrainedTokenizer:
        """Creates the tokenizer."""
        raise NotImplementedError("create_tokenizer method must be implemented")
