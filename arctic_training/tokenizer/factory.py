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
from typing import Type

from transformers import PreTrainedTokenizer

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.tokenizer import TokenizerConfig

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


class TokenizerFactory(ABC, CallbackMixin):
    name: str
    config_type: Type[TokenizerConfig] = TokenizerConfig

    def __init__(
        self, trainer: "Trainer", tokenizer_config: Optional["TokenizerConfig"] = None
    ) -> None:
        if tokenizer_config is None:
            tokenizer_config = trainer.config.tokenizer

        self.trainer = trainer
        self.config = tokenizer_config

    def __call__(self) -> PreTrainedTokenizer:
        tokenizer = self.create_tokenizer()
        return tokenizer

    @abstractmethod
    @callback_wrapper("create-tokenizer")
    def create_tokenizer(self) -> PreTrainedTokenizer:
        raise NotImplementedError("create_tokenizer method must be implemented")
