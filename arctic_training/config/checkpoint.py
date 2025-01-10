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

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Type

from pydantic import Field

from arctic_training.config import BaseConfig
from arctic_training.registry.checkpoint import get_registered_checkpoint_engine

if TYPE_CHECKING:
    from arctic_training.checkpoint.engine import CheckpointEngine


class CheckpointConfig(BaseConfig):
    type: str = ""
    """ Checkpoint engine type. """

    output_dir: Path
    """ Checkpoint output directory. If directory does not exist, it will be created. """

    enabled: bool = True
    """ Enable this checkpoint engine. """

    auto_resume: bool = False
    """ If a checkpoint is found in the output directory, resume training from that checkpoint. """

    save_every_n_steps: int = Field(default=0, ge=0)
    """ How often to trigger a checkpoint save by training global step count. """

    save_every_n_epochs: int = Field(default=0, ge=0)
    """ How often to trigger a checkpoint save by training epoch count. """

    save_end_of_training: bool = False
    """ Whether to save a checkpoint at the end of training. """

    @property
    def engine(self) -> Type["CheckpointEngine"]:
        return get_registered_checkpoint_engine(self.type)
