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

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.optimizer import OptimizerConfig

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer


class OptimizerFactory(ABC, CallbackMixin):
    name: str
    config_type: Type[OptimizerConfig] = OptimizerConfig

    def __init__(
        self,
        trainer: "Trainer",
        optimizer_config: Optional["OptimizerConfig"] = None,
    ) -> None:
        if optimizer_config is None:
            optimizer_config = trainer.config.optimizer

        self.trainer = trainer
        self.config = optimizer_config

    def __call__(self) -> Any:
        optimizer = self.create_optimizer(self.trainer.model, self.config)
        return optimizer

    @abstractmethod
    @callback_wrapper("create-optimizer")
    def create_optimizer(self, model: Any, optimizer_config: "OptimizerConfig") -> Any:
        pass
