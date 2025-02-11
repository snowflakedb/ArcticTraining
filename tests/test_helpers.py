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

from datasets import Dataset
from datasets import load_dataset
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel

from arctic_training import register
from arctic_training.config.data import DataConfig
from arctic_training.config.optimizer import OptimizerConfig
from arctic_training.config.scheduler import SchedulerConfig
from arctic_training.data.factory import DataFactory
from arctic_training.data.source import DataSource
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.optimizer.factory import OptimizerFactory
from arctic_training.scheduler.factory import SchedulerFactory


@register
class RandomWeightHFModelFactory(HFModelFactory):
    name = "random-weight-hf"

    def create_model(self, model_config) -> PreTrainedModel:
        return AutoModelForCausalLM.from_config(
            model_config,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=self.config.dtype,
        )


@register
class UltraChat200KTruncated(DataSource):
    name = "HuggingFaceH4/ultrachat_200k-truncated"
    data_factory_type = "sft"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        streamed_data = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="test_sft" if eval else "train_sft",
            streaming=True,
        )
        subset = Dataset.from_list(
            list(streamed_data.take(20)), features=streamed_data.features
        )
        return subset.select_columns(["messages"])


@register
class CPUAdamOptimizerFactory(FusedAdamOptimizerFactory):
    name = "cpu-adam"

    def create_optimizer(self, model, optimizer_config):
        optimizer_grouped_params = self.get_optimizer_grouped_params(
            model, optimizer_config.weight_decay
        )
        return DeepSpeedCPUAdam(
            optimizer_grouped_params,
            lr=optimizer_config.learning_rate,
            betas=optimizer_config.betas,
        )


@register
class NoOpOptimizerFactory(OptimizerFactory):
    name = "noop"
    config_type = OptimizerConfig

    def create_optimizer(self, model, optimizer_config):
        return None


@register
class NoOpDataFactory(DataFactory):
    name = "noop"
    config_type = DataConfig

    def __call__(self):
        return None, None

    def tokenize_fn(self):
        pass

    def collate_fn(self):
        pass


@register
class NoOpSchedulerFactory(SchedulerFactory):
    name = "noop"
    config_type = SchedulerConfig

    def create_scheduler(self, optimizer):
        return None
