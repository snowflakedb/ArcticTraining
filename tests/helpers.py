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
from datasets import IterableDataset
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel

from arctic_training.data.factory import DataFactory
from arctic_training.data.hf_source import SlimOrca
from arctic_training.data.hf_source import UltraChat200K
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.scheduler.factory import SchedulerFactory


class RandomWeightHFModelFactory(HFModelFactory):
    name = "random-weight-hf"

    def create_model(self, model_config) -> PreTrainedModel:
        return AutoModelForCausalLM.from_config(
            model_config,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=self.config.dtype,
        )


def modify_config_for_truncated_data(self):
    self.config.kwargs["streaming"] = True  # Avoid downloading entire dataset
    self.config.dataset_name = self.name.removesuffix(  # Set to the real dataset name
        "-truncated"
    )


def sample_data_for_truncated_dataset(self, dataset: IterableDataset) -> Dataset:
    return Dataset.from_list(list(dataset.take(20)), features=dataset.features)


class UltraChat200KTruncated(UltraChat200K):
    name = "HuggingFaceH4/ultrachat_200k-truncated"
    callbacks = [
        ("post-init", modify_config_for_truncated_data),
        ("post-load", sample_data_for_truncated_dataset),
    ]


class SlimOrcaTruncated(SlimOrca):
    name = "Open-Orca/SlimOrca-truncated"
    callbacks = [
        ("post-init", modify_config_for_truncated_data),
        ("post-load", sample_data_for_truncated_dataset),
    ]


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


class NoOpDataFactory(DataFactory):
    name = "noop"

    def __call__(self):
        return None, None

    def tokenize(self, tokenizer, dataset):
        return dataset


class NoOpSchedulerFactory(SchedulerFactory):
    name = "noop"

    def create_scheduler(self, optimizer):
        return None
