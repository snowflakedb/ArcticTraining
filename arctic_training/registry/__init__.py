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

from typing import Type
from typing import Union

from .checkpoint import register_checkpoint_engine
from .data import register_data_factory
from .data import register_data_source
from .model import register_model_factory
from .optimizer import register_optimizer_factory
from .scheduler import register_scheduler_factory
from .tokenizer import register_tokenizer_factory
from .trainer import register_trainer
from .utils import RegistryClassTypes


def register(cls: RegistryClassTypes, force: bool = False) -> RegistryClassTypes:
    from arctic_training.checkpoint.engine import CheckpointEngine
    from arctic_training.data.factory import DataFactory
    from arctic_training.data.source import DataSource
    from arctic_training.model.factory import ModelFactory
    from arctic_training.optimizer.factory import OptimizerFactory
    from arctic_training.scheduler.factory import SchedulerFactory
    from arctic_training.tokenizer.factory import TokenizerFactory
    from arctic_training.trainer.trainer import Trainer

    if issubclass(cls, CheckpointEngine):
        return register_checkpoint_engine(cls, force)
    elif issubclass(cls, DataFactory):
        return register_data_factory(cls, force)
    elif issubclass(cls, DataSource):
        return register_data_source(cls, force)
    elif issubclass(cls, ModelFactory):
        return register_model_factory(cls, force)
    elif issubclass(cls, OptimizerFactory):
        return register_optimizer_factory(cls, force)
    elif issubclass(cls, SchedulerFactory):
        return register_scheduler_factory(cls, force)
    elif issubclass(cls, TokenizerFactory):
        return register_tokenizer_factory(cls, force)
    elif issubclass(cls, Trainer):
        return register_trainer(cls, force)
    else:
        raise ValueError(f"Unsupported class type for registration: {cls.__name__}")
