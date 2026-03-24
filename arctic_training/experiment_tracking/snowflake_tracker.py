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

from arctic_training.config.experiment_tracking import ExperimentTrackingConfig
from arctic_training.experiment_tracking.tracker import ExperimentTracker
from arctic_training.logging import logger

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


class SnowflakeExperimentTrackingConfig(ExperimentTrackingConfig):
    type: str = "snowflake"

    account: str = ""
    """ Snowflake account identifier. """

    user: str = ""
    """ Snowflake user name. """

    password: str = ""
    """ Snowflake PAT or password. """

    role: str = ""
    """ Snowflake role. """

    warehouse: str = ""
    """ Snowflake warehouse. """

    database: str = ""
    """ Snowflake database. """

    schema_name: str = ""
    """ Snowflake schema. """

    experiment_name: str = ""
    """ Name of the experiment in Snowflake experiment tracking. """

    run_name: Optional[str] = None
    """ Name of the run. If not set, one will be generated. """

    @property
    def connection_params(self):
        return dict(
            account=self.account,
            user=self.user,
            authentication="PAT",
            password=self.password,
            role=self.role,
            warehouse=self.warehouse,
            database=self.database,
            schema=self.schema_name,
        )


class SnowflakeExpTracker(ExperimentTracker):
    name: str = "snowflake"
    config: SnowflakeExperimentTrackingConfig

    def __init__(self, trainer: "Trainer", config: SnowflakeExperimentTrackingConfig) -> None:
        super().__init__(trainer, config)
        self._experiment: Any = None
        self._run_name: Optional[str] = config.run_name

    def start(self, run_config: Dict[str, Any]) -> None:
        try:
            from snowflake.ml.experiment.experiment_tracking import ExperimentTracking
            from snowflake.snowpark import Session
        except ImportError:
            raise ImportError(
                "Snowflake experiment tracking requires the snowflake-ml-python and "
                "snowflake-snowpark-python packages. Install them with:\n"
                "  pip install snowflake-ml-python snowflake-snowpark-python"
            )

        session = Session.builder.configs(self.config.connection_params).create()
        self._experiment = ExperimentTracking(session=session)
        self._experiment.set_experiment(self.config.experiment_name)

        self._experiment.start_run(self._run_name)
        self._experiment.log_params(run_config)
        logger.info(f"Snowflake experiment tracking started: {self.config.experiment_name}/{self._run_name}")

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if self._experiment is not None:
            self._experiment.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        if self._experiment is not None:
            self._experiment.log_params(params)

    def finish(self) -> None:
        if self._experiment is not None:
            self._experiment.end_run()

    def get_resume_state(self) -> Optional[Dict[str, Any]]:
        if self._run_name is None:
            return None
        return {"run_name": self._run_name}

    def set_resume_state(self, state: Dict[str, Any]) -> None:
        self._run_name = state.get("run_name")
