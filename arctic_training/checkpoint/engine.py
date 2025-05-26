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

import torch

from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.checkpoint import CheckpointConfig
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


class CheckpointEngine(ABC, CallbackMixin, metaclass=RegistryMeta):
    """Base class for all checkpoint engines."""

    name: str
    """
    The name of the checkpoint engine. This is used to identify the checkpoint
    engine in the registry.
    """

    config: CheckpointConfig
    """
    The configuration class for the checkpoint engine. This is used to validate
    the configuration passed to the engine.
    """

    @classmethod
    def _validate_subclass(cls) -> None:
        _validate_class_attribute_set(cls, "name")
        _validate_class_attribute_type(cls, "config", CheckpointConfig)
        _validate_class_method(cls, "load", ["self", "model"])
        _validate_class_method(cls, "save", ["self", "model"])

    def __init__(self, trainer: "Trainer", config: CheckpointConfig) -> None:
        self._trainer = trainer
        self.config = config

    @property
    def trainer(self) -> "Trainer":
        return self._trainer

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
    def epoch_finished(self) -> bool:
        return self.trainer.epoch_finished

    @property
    def training_finished(self) -> bool:
        return self.trainer.training_finished

    @property
    def do_checkpoint(self) -> bool:
        """
        Checks the current state of the trainer and determines if we are at a
        checkpoint boundary.
        """
        if not self.config.enabled:
            return False
        return_value = False
        if (
            self.trainer.model.is_gradient_accumulation_boundary()
            and self.config.save_every_n_steps
            and self.trainer.global_step > 0
        ):
            return_value = self.trainer.global_step % self.config.save_every_n_steps == 0
        if self.config.save_every_n_epochs:
            return_value = return_value or (
                self.epoch_finished and (self.trainer.epoch_idx % self.config.save_every_n_epochs) == 0
            )
        if self.config.save_end_of_training:
            return_value = return_value or self.training_finished
        return return_value

    @property
    def checkpoint_dir(self) -> Path:
        """Returns the directory where the checkpoint will be saved."""
        checkpoint_dir = self.config.output_dir / f"global_step_{self.trainer.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    @abstractmethod
    @callback_wrapper("load")
    def load(self, model: Any) -> Any:
        """Loads the model weights from a checkpoint when training is resumed."""
        raise NotImplementedError("load method must be implemented")

    @abstractmethod
    @callback_wrapper("save")
    def save(self, model: Any) -> None:
        """Saves the model weights to a checkpoint."""
        raise NotImplementedError("save method must be implemented")
