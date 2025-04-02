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

from transformers import PreTrainedModel

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.model import ModelConfig
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


class ModelFactory(ABC, CallbackMixin, metaclass=RegistryMeta):
    """Base class for model creation."""

    name: str
    """
    Name of the model factory used for registering custom model factories. This
    name should be unique and is used in training recipe YAMLs to identify which
    model factory to be used.
    """

    config: ModelConfig
    """
    The type of config class that the model factory uses. This should contain
    all model-specific parameters.
    """

    @classmethod
    def _validate_subclass(cls) -> None:
        _validate_class_attribute_set(cls, "name")
        _validate_class_attribute_type(cls, "config", ModelConfig)
        _validate_class_method(cls, "create_config", ["self"])
        _validate_class_method(cls, "create_model", ["self", "model_config"])

    def __init__(self, trainer: "Trainer", model_config: Optional[ModelConfig] = None) -> None:
        if model_config is None:
            model_config = trainer.config.model

        self._trainer = trainer
        self.config = model_config

    def __call__(self) -> PreTrainedModel:
        config = self.create_config()
        model = self.create_model(model_config=config)
        return model

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
    @callback_wrapper("create-config")
    def create_config(self) -> Any:
        """Creates the model config (e.g., huggingface model config)."""
        raise NotImplementedError("create_config method must be implemented")

    @abstractmethod
    @callback_wrapper("create-model")
    def create_model(self, model_config) -> PreTrainedModel:
        """Creates the model."""
        raise NotImplementedError("create_model method must be implemented")
