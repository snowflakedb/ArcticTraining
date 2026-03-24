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
from typing import Dict
from typing import Optional

from arctic_training.config.experiment_tracking import ExperimentTrackingConfig
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


class ExperimentTracker(ABC, metaclass=RegistryMeta):
    """Base class for experiment tracking backends."""

    name: str
    """
    Name of the experiment tracker used for registration. This name
    should be unique and is used in configs to select the tracker.
    """

    config: ExperimentTrackingConfig
    """
    The type of the config class that the tracker uses. This should be a
    subclass of ExperimentTrackingConfig.
    """

    @classmethod
    def _validate_subclass(cls) -> None:
        _validate_class_attribute_set(cls, "name")
        _validate_class_attribute_type(cls, "config", ExperimentTrackingConfig)
        _validate_class_method(cls, "log_metrics", ["self", "metrics", "step"])
        _validate_class_method(cls, "log_params", ["self", "params"])
        _validate_class_method(cls, "finish", ["self"])
        _validate_class_method(cls, "get_resume_state", ["self"])
        _validate_class_method(cls, "set_resume_state", ["self", "state"])

    def __init__(self, trainer: "Trainer", config: ExperimentTrackingConfig) -> None:
        self._trainer = trainer
        self.config = config

    @property
    def trainer(self) -> "Trainer":
        return self._trainer

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics for the current step."""
        raise NotImplementedError

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters/config for the run."""
        raise NotImplementedError

    @abstractmethod
    def finish(self) -> None:
        """Finalize and close the tracking run."""
        raise NotImplementedError

    @abstractmethod
    def get_resume_state(self) -> Optional[Dict[str, Any]]:
        """Return state needed to resume this tracker across checkpoint restarts."""
        raise NotImplementedError

    @abstractmethod
    def set_resume_state(self, state: Dict[str, Any]) -> None:
        """Restore tracker state from a previous checkpoint."""
        raise NotImplementedError
