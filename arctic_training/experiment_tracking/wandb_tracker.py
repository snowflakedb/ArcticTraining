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
from typing import Any
from typing import Dict
from typing import Optional

import wandb

from arctic_training.config.experiment_tracking import ExperimentTrackingConfig
from arctic_training.experiment_tracking.tracker import ExperimentTracker

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


class WandBExperimentTrackingConfig(ExperimentTrackingConfig):
    type: str = "wandb"

    entity: Optional[str] = None
    """ Weights and Biases entity name. """

    project: Optional[str] = "arctic-training"
    """ Weights and Biases project name. """

    name: Optional[str] = None
    """ Weights and Biases run name. """


class WandBTracker(ExperimentTracker):
    name: str = "wandb"
    config: WandBExperimentTrackingConfig

    def __init__(self, trainer: "Trainer", config: WandBExperimentTrackingConfig) -> None:
        super().__init__(trainer, config)
        self._run_id: Optional[str] = None
        self._run = None

    def start(self, run_config: Dict[str, Any]) -> None:
        if self._run_id is None:
            self._run_id = wandb.util.generate_id()

        self._run = wandb.init(
            id=self._run_id,
            entity=self.config.entity,
            project=self.config.project,
            name=self.config.name,
            config=run_config,
            dir=f"{self.trainer.config.logger.output_dir}/wandb",
        )

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(params)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()

    def get_resume_state(self) -> Optional[Dict[str, Any]]:
        if self._run_id is None:
            return None
        return {"wandb_run_id": self._run_id}

    def set_resume_state(self, state: Dict[str, Any]) -> None:
        self._run_id = state.get("wandb_run_id")
