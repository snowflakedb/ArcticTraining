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
from typing import Literal

from torch.optim.lr_scheduler import LambdaLR
from transformers import get_wsd_schedule

from arctic_training.config.scheduler import SchedulerConfig
from arctic_training.scheduler.factory import SchedulerFactory


class WSDSchedulerConfig(SchedulerConfig):
    """See: https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules#transformers.get_wsd_schedule"""

    name: str = "wsd"
    num_warmup_steps: int
    num_decay_steps: int
    warmup_type: Literal["linear", "cosine", "1-sqrt"] = "linear"
    decay_type: Literal["linear", "cosine", "1-sqrt"] = "linear"
    min_lr_ratio: float = 0.0
    num_cycles: float = 0.5


class WSDSchedulerFactory(SchedulerFactory):
    name = "wds"
    config: WSDSchedulerConfig

    def create_scheduler(self, optimizer: Any) -> LambdaLR:
        return get_wsd_schedule(
            name=self.config.name,
            optimizer=optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_decay_steps=self.config.num_decay_steps,
            warmup_type=self.config.warmup_type,
            decay_type=self.config.decay_type,
            min_lr_ratio=self.config.min_lr_ratio,
            num_cycles=self.config.num_cycles,
            num_training_steps=self.trainer.training_horizon,
        )
