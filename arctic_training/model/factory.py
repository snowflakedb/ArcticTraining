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
from typing import Any
from typing import Optional
from typing import Type

from transformers import PreTrainedModel

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.model import ModelConfig

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer


class ModelFactory(ABC, CallbackMixin):
    name: str
    config_type: Type[ModelConfig] = ModelConfig

    def __init__(
        self, trainer: "Trainer", model_config: Optional["ModelConfig"] = None
    ) -> None:
        if model_config is None:
            model_config = trainer.config.model

        self.trainer = trainer
        self.config = model_config

    def __call__(self) -> PreTrainedModel:
        config = self.create_config()
        model = self.create_model(model_config=config)
        return model

    @abstractmethod
    @callback_wrapper("create-config")
    def create_config(self) -> Any:
        raise NotImplementedError("create_config method must be implemented")

    @abstractmethod
    @callback_wrapper("create-model")
    def create_model(self, model_config) -> PreTrainedModel:
        raise NotImplementedError("create_model method must be implemented")
