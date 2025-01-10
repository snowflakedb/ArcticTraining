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
from typing import List
from typing import Literal
from typing import Union

from pydantic import field_validator

from arctic_training.logging import LOG_LEVEL_DEFAULT
from arctic_training.utils import get_local_rank
from arctic_training.utils import get_world_size

from .base import BaseConfig


class LoggerConfig(BaseConfig):
    output_dir: Path = Path("/dev/null")
    """ Output directory for log files. """

    level: str = LOG_LEVEL_DEFAULT
    """ Log level for the logger. """

    print_output_ranks: Union[Literal["*"], List[int]] = [0]
    """ Which ranks will print logs. Either a list of ranks or "*" for all ranks. """

    file_output_ranks: Union[Literal["*"], List[int]] = "*"
    """ Which ranks will output logs to a file. Either a list of ranks or "*" for all ranks. """

    @field_validator("print_output_ranks", "file_output_ranks", mode="before")
    def fill_output_ranks(cls, v):
        if v == "*":
            return range(get_world_size())
        return v

    @property
    def log_file(self) -> Path:
        local_rank = get_local_rank()
        return self.output_dir / f"rank_{local_rank}.log"

    @property
    def file_enabled(self) -> bool:
        if self.output_dir == Path("/dev/null"):
            return False
        local_rank = get_local_rank()
        return local_rank in self.file_output_ranks

    @property
    def print_enabled(self) -> bool:
        local_rank = get_local_rank()
        return local_rank in self.print_output_ranks
