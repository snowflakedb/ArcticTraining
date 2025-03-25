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

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from arctic_training.config.utils import get_global_rank
from arctic_training.config.utils import get_local_rank
from arctic_training.config.utils import get_world_size
from arctic_training.logging import logger


class BaseConfig(BaseModel):
    def __init__(self, **data):
        logger.info(f"Initializing {self.__class__.__name__}")
        super().__init__(**data)

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        # validate_assignment=True,
        validate_default=True,
        use_attribute_docstrings=True,
        populate_by_name=True,
    )

    local_rank: int = Field(default_factory=get_local_rank, exclude=True)
    global_rank: int = Field(default_factory=get_global_rank, exclude=True)
    world_size: int = Field(default_factory=get_world_size, exclude=True)
