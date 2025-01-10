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

from typing import Dict

from arctic_training.checkpoint.ds_engine import DSCheckpointEngine
from arctic_training.checkpoint.hf_engine import HFCheckpointEngine
from arctic_training.config.trainer import TrainerConfig
from arctic_training.data.sft_factory import SFTDataFactory
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.model.liger_factory import LigerModelFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.registry import register
from arctic_training.scheduler.hf_factory import HFSchedulerFactory
from arctic_training.tokenizer.hf_factory import HFTokenizerFactory
from arctic_training.trainer import Trainer


def to_device(batch: Dict, device: str) -> Dict:
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except Exception:
            output[k] = v
    return output


@register
class SFTTrainer(Trainer):
    name = "sft"
    config_type = TrainerConfig
    data_factory_type = [SFTDataFactory]
    model_factory_type = [HFModelFactory, LigerModelFactory]
    checkpoint_engine_type = [DSCheckpointEngine, HFCheckpointEngine]
    optimizer_factory_type = [FusedAdamOptimizerFactory]
    scheduler_factory_type = [HFSchedulerFactory]
    tokenizer_factory_type = [HFTokenizerFactory]

    def loss(self, batch):
        batch = to_device(batch, self.device)
        outputs = self.model(**batch, use_cache=False)
        loss = outputs.loss
        return loss
