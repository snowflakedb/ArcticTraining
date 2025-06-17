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

from typing import Any

from pydantic import Field
from transformers import get_scheduler

from arctic_training.config.scheduler import SchedulerConfig
from arctic_training.config.utils import HumanFloat
from arctic_training.scheduler.factory import SchedulerFactory


class HFSchedulerConfig(SchedulerConfig):
    name: str = "linear"
    warmup_ratio: HumanFloat = Field(default=0.1, ge=0.0, le=1.0)
    """ The fraction of total training steps used for linear learning rate warmup. """


class HFSchedulerFactory(SchedulerFactory):
    name = "huggingface"
    config: HFSchedulerConfig

    def create_scheduler(self, optimizer: Any) -> Any:
        num_warmup_steps = int(self.config.warmup_ratio * self.trainer.training_horizon)
        return get_scheduler(
            name=self.config.name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps * self.trainer.config.sequence_parallel_size,
            num_training_steps=self.trainer.training_horizon * self.trainer.config.sequence_parallel_size,
        )
