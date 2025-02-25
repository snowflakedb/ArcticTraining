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

from typing import Optional

from arctic_training.config.base import BaseConfig


class WandBConfig(BaseConfig):
    enable: bool = False
    """ Whether to enable Weights and Biases logging. """

    entity: Optional[str] = None
    """ Weights and Biases entity name. """

    project: Optional[str] = "arctic-training"
    """ Weights and Biases project name. """

    name: Optional[str] = None
    """ Weights and Biases run name. """
