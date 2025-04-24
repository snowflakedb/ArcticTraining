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
from typing import Tuple
from typing import Type

from pydantic import Field

from arctic_training.config.base import BaseConfig
from arctic_training.config.utils import HumanFloat
from arctic_training.registry import get_registered_optimizer_factory

if TYPE_CHECKING:
    from arctic_training.optimizer.factory import OptimizerFactory


class OptimizerConfig(BaseConfig):
    type: str = ""
    """ Optimizer factory type. Defaults to the `optimizer_factory_type` of the trainer. """

    weight_decay: HumanFloat = Field(default=0.1, ge=0.0)
    """ Coefficient for L2 regularization applied to the optimizer's weights. """

    betas: Tuple[float, float] = (0.9, 0.999)
    """ Tuple of coefficients used for computing running averages of gradient and its square (e.g., (beta1, beta2) for Adam). """

    learning_rate: HumanFloat = Field(default=5e-4, ge=0.0, alias="lr")
    """ The initial learning rate. """

    @property
    def factory(self) -> Type["OptimizerFactory"]:
        return get_registered_optimizer_factory(name=self.type)
