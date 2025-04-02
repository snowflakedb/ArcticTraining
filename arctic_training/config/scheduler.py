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
from typing import Optional
from typing import Type

from pydantic import Field
from pydantic import field_validator

from arctic_training.config.base import BaseConfig
from arctic_training.registry import get_registered_scheduler_factory

if TYPE_CHECKING:
    from arctic_training.scheduler.factory import SchedulerFactory


class SchedulerConfig(BaseConfig):
    type: str = ""
    """ Scheduler factory type. Defaults to the `scheduler_factory_type` of the trainer. """

    learning_rate: Optional[float] = Field(default=None, alias="lr")
    """ The initial learning rate. Deprecated in favor of `optimizer.learning_rate`. """

    @field_validator("learning_rate", mode="after")
    @classmethod
    def _deprecated_learning_rate(cls, value: Optional[float]) -> Optional[float]:
        if value is not None:
            raise ValueError("scheduler.learning_rate is deprecated. Use optimizer.learning_rate instead.")
        return value

    @property
    def factory(self) -> Type["SchedulerFactory"]:
        return get_registered_scheduler_factory(name=self.type)
