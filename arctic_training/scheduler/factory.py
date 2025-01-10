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
from typing import Type

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.scheduler import SchedulerConfig

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer


class SchedulerFactory(ABC, CallbackMixin):
    name: str
    config_type: Type[SchedulerConfig] = SchedulerConfig

    def __init__(
        self,
        trainer: "Trainer",
        scheduler_config=None,
    ) -> None:
        if scheduler_config is None:
            scheduler_config = trainer.config.scheduler

        self.trainer = trainer
        self.config = scheduler_config

    def __call__(self) -> Any:
        scheduler = self.create_scheduler(optimizer=self.trainer.optimizer)
        return scheduler

    @abstractmethod
    @callback_wrapper("create-scheduler")
    def create_scheduler(self, optimizer: Any) -> Any:
        raise NotImplementedError
