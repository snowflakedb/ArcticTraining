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
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Type

import torch

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.checkpoint import CheckpointConfig

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer


class CheckpointEngine(ABC, CallbackMixin):
    name: str
    config_type: Type[CheckpointConfig] = CheckpointConfig

    def __init__(self, trainer: "Trainer", config: "CheckpointConfig") -> None:
        self.trainer = trainer
        self.config = config

    @property
    def global_rank(self) -> int:
        return self.trainer.global_rank

    @property
    def world_size(self) -> int:
        return self.trainer.world_size

    @property
    def device(self) -> torch.device:
        return self.trainer.device

    @property
    def training_finished(self) -> bool:
        return self.trainer.training_finished

    @property
    def do_checkpoint(self) -> bool:
        if not self.config.enabled:
            return False
        if (
            self.trainer.model.is_gradient_accumulation_boundary()
            and self.config.save_every_n_steps
            and self.trainer.global_step > 0
        ):
            return self.trainer.global_step % self.config.save_every_n_steps == 0
        if self.config.save_every_n_epochs:
            return (self.trainer.epoch_idx > 0) and (
                self.trainer.epoch_idx % self.config.save_every_n_epochs
            ) == 0
        if self.config.save_end_of_training:
            return self.training_finished
        return False

    @property
    def checkpoint_dir(self) -> Path:
        checkpoint_dir = (
            self.config.output_dir / f"global_step_{self.trainer.global_step}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    @abstractmethod
    @callback_wrapper("load")
    def load(self, model: Any) -> Any:
        raise NotImplementedError("load method must be implemented")

    @abstractmethod
    @callback_wrapper("save")
    def save(self, model: Any) -> None:
        raise NotImplementedError("save method must be implemented")
