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

import importlib
import typing as t

from arctic_training.logging import setup_init_logger

setup_init_logger()

__all__ = [
    # Public symbols exposed by the package
    "logger",
    "register",
    "Callback",
    "DSCheckpointEngine",
    "CheckpointEngine",
    "HFCheckpointEngine",
    "CheckpointConfig",
    "DataConfig",
    "LoggerConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TokenizerConfig",
    "TrainerConfig",
    "get_config",
    "DPODataFactory",
    "DataFactory",
    "HFDataSource",
    "SFTDataFactory",
    "DataSource",
    "ModelFactory",
    "HFModelFactory",
    "LigerModelFactory",
    "FusedAdamOptimizerFactory",
    "OptimizerFactory",
    "SchedulerFactory",
    "HFSchedulerFactory",
    "TokenizerFactory",
    "HFTokenizerFactory",
    "DPOTrainer",
    "DPOTrainerConfig",
    "SFTTrainer",
    "Trainer",
]

_import_map = {
    "logger": "arctic_training.logging",
    "register": "arctic_training.registry",
    "Callback": "arctic_training.callback.callback",
    "DSCheckpointEngine": "arctic_training.checkpoint.ds_engine",
    "CheckpointEngine": "arctic_training.checkpoint.engine",
    "HFCheckpointEngine": "arctic_training.checkpoint.hf_engine",
    "CheckpointConfig": "arctic_training.config.checkpoint",
    "DataConfig": "arctic_training.config.data",
    "LoggerConfig": "arctic_training.config.logger",
    "ModelConfig": "arctic_training.config.model",
    "OptimizerConfig": "arctic_training.config.optimizer",
    "SchedulerConfig": "arctic_training.config.scheduler",
    "TokenizerConfig": "arctic_training.config.tokenizer",
    "TrainerConfig": "arctic_training.config.trainer",
    "get_config": "arctic_training.config.trainer",
    "DataFactory": "arctic_training.data.factory",
    "DPODataFactory": "arctic_training.data.dpo_factory",
    "DataSource": "arctic_training.data.source",
    "HFDataSource": "arctic_training.data.hf_source",
    "SFTDataFactory": "arctic_training.data.sft_factory",
    "ModelFactory": "arctic_training.model.factory",
    "HFModelFactory": "arctic_training.model.hf_factory",
    "LigerModelFactory": "arctic_training.model.liger_factory",
    "OptimizerFactory": "arctic_trainng.optimizer.factory",
    "FusedAdamOptimizerFactory": "arctic_training.optimizer.adam_factory",
    "SchedulerFactory": "arctic_training.scheduler.factory",
    "HFSchedulerFactory": "arctic_training.scheduler.hf_factory",
    "TokenizerFactory": "arctic_training.tokenizer.factory",
    "HFTokenizerFactory": "arctic_training.tokenizer.hf_factory",
    "Trainer": "arctic_training.trainer.trainer",
    "DPOTrainer": "arctic_training.trainer.dpo_trainer",
    "DPOTrainerConfig": "arctic_training.trainer.dpo_trainer",
    "SFTTrainer": "arctic_training.trainer.sft_trainer",
}

_lazy_cache: dict[str, t.Any] = {}


def __getattr__(name: str) -> t.Any:
    if name in _lazy_cache:
        return _lazy_cache[name]
    if name in _import_map:
        module = importlib.import_module(_import_map[name])
        attr = getattr(module, name)
        _lazy_cache[name] = attr
        return attr
    raise AttributeError(f"module {__name__} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_import_map.keys()))
