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
from typing import Type

from pydantic import Field

from arctic_training.config.base import BaseConfig
from arctic_training.registry.scheduler import get_registered_scheduler_factory

if TYPE_CHECKING:
    from arctic_training.scheduler.factory import SchedulerFactory


class SchedulerConfig(BaseConfig):
    type: str = ""
    """ Scheduler factory type. Defaults to the `scheduler_factory_type` of the trainer. """

    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    """ The fraction of total training steps used for linear learning rate warmup. """

    learning_rate: float = Field(default=5e-4, ge=0.0, alias="lr")
    """ The initial learning rate. """

    @property
    def factory(self) -> Type["SchedulerFactory"]:
        return get_registered_scheduler_factory(self.type)
