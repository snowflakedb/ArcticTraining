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

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.optimizer import OptimizerConfig
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


class OptimizerFactory(ABC, CallbackMixin, metaclass=RegistryMeta):
    """Base class for optimizer creation."""

    name: str
    """
    Name of the optimizer factory used for registering custom optimizer
    factories. This name should be unique and is used in training recipe YAMLs
    to identify which optimizer factory to be used.
    """

    config: OptimizerConfig
    """
    The type of config class that the optimizer factory uses. This should
    contain all optimizer-specific parameters.
    """

    @classmethod
    def _validate_subclass(cls) -> None:
        _validate_class_attribute_set(cls, "name")
        _validate_class_attribute_type(cls, "config", OptimizerConfig)
        _validate_class_method(cls, "create_optimizer", ["self", "model", "optimizer_config"])

    def __init__(
        self,
        trainer: "Trainer",
        optimizer_config: Optional[OptimizerConfig] = None,
    ) -> None:
        if optimizer_config is None:
            optimizer_config = trainer.config.optimizer

        self._trainer = trainer
        self.config = optimizer_config

    def __call__(self) -> Any:
        optimizer = self.create_optimizer(self.trainer.model, self.config)
        return optimizer

    @property
    def trainer(self) -> "Trainer":
        return self._trainer

    @property
    def device(self) -> str:
        return self.trainer.device

    @property
    def model(self) -> Any:
        return self.trainer.model

    @property
    def world_size(self) -> int:
        return self.trainer.world_size

    @property
    def global_rank(self) -> int:
        return self.trainer.global_rank

    @abstractmethod
    @callback_wrapper("create-optimizer")
    def create_optimizer(self, model: Any, optimizer_config: "OptimizerConfig") -> Any:
        """Creates the optimizer given a model and an optimizer config."""
        pass
