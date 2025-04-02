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

from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam

if TYPE_CHECKING:
    from arctic_training.config.optimizer import OptimizerConfig

from typing import Dict
from typing import List

from transformers import PreTrainedModel

from arctic_training.optimizer.factory import OptimizerFactory


class FusedAdamOptimizerFactory(OptimizerFactory):
    name = "fusedadam"

    @staticmethod
    def get_optimizer_grouped_params(
        model: PreTrainedModel,
        weight_decay: float,
        no_decay_name_list: List[str] = [
            "bias",
            "layer_norm.weight",
            "layernorm.weight",
            "norm.weight",
            "ln_f.weight",
        ],
    ) -> List[Dict]:
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (not any(nd in n.lower() for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (any(nd in n.lower() for nd in no_decay_name_list) and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        non_empty_groups = []
        for group in optimizer_grouped_parameters:
            if group["params"]:
                non_empty_groups.append(group)
        return non_empty_groups

    def create_optimizer(self, model: Any, optimizer_config: "OptimizerConfig") -> Any:
        optimizer_grouped_params = self.get_optimizer_grouped_params(model, optimizer_config.weight_decay)
        optimizer = FusedAdam(
            optimizer_grouped_params,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
        )
        return optimizer


class CPUAdamOptimizerFactory(FusedAdamOptimizerFactory):
    name = "cpu_adam"

    def create_optimizer(self, model: Any, optimizer_config: "OptimizerConfig") -> Any:
        optimizer_grouped_params = self.get_optimizer_grouped_params(model, optimizer_config.weight_decay)
        optimizer = DeepSpeedCPUAdam(
            optimizer_grouped_params,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
        )
        return optimizer
