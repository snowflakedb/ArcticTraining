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

import random
from abc import ABC
from typing import TYPE_CHECKING
from typing import Any
from typing import List
from typing import Type

import deepspeed
import numpy as np
import torch
from devtools import debug
from tqdm import tqdm
from transformers import set_seed

from arctic_training.callback.logging import post_loss_log_cb
from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.config.trainer import TrainerConfig
from arctic_training.logging import logger

if TYPE_CHECKING:
    from arctic_training.checkpoint.engine import CheckpointEngine
    from arctic_training.data.factory import DataFactory
    from arctic_training.model.factory import ModelFactory
    from arctic_training.optimizer.factory import OptimizerFactory
    from arctic_training.scheduler.factory import SchedulerFactory
    from arctic_training.tokenizer.factory import TokenizerFactory

try:
    from transformers.integrations.deepspeed import HfDeepSpeedConfig
except ImportError:
    from transformers.deepspeed import HfDeepSpeedConfig


class Trainer(ABC, CallbackMixin):
    name: str

    config_type: Type[TrainerConfig] = TrainerConfig
    data_factory_type: List[Type["DataFactory"]]
    model_factory_type: List[Type["ModelFactory"]]
    checkpoint_engine_type: List[Type["CheckpointEngine"]]
    optimizer_factory_type: List[Type["OptimizerFactory"]]
    scheduler_factory_type: List[Type["SchedulerFactory"]]
    tokenizer_factory_type: List[Type["TokenizerFactory"]]

    callbacks = [post_loss_log_cb]

    def __init__(self, config: "TrainerConfig") -> None:
        logger.info(f"Initializing Trainer with config:\n{debug.format(config)}")
        self.config = config
        self.epoch_idx = 0
        self.train_batch_idx = 0
        self.global_step = 0
        self.eval_batch_idx = 0
        self.early_stop = False
        self.world_size = config.world_size
        self.global_rank = config.global_rank
        self.training_finished = False

        self._set_seeds(self.config.seed)

        tokenizer_factory = self.config.tokenizer.factory(self)
        self.tokenizer = tokenizer_factory()

        data_factory = self.config.data.factory(self)
        self.train_dataloader, self.eval_dataloader = data_factory()

        dschf = HfDeepSpeedConfig(self.config.deepspeed)  # noqa: F841
        model_factory = self.config.model.factory(self)
        self.model = model_factory()

        optimizer_factory = self.config.optimizer.factory(self)
        self.optimizer = optimizer_factory()

        scheduler_factory = self.config.scheduler.factory(self)
        self.scheduler = scheduler_factory()

        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            args=self.config,
            lr_scheduler=self.scheduler,
            config=self.config.deepspeed,
        )

        self.checkpoint_engines = [
            engine(self) for engine in self.config.checkpoint_engines
        ]

    def _set_seeds(self, seed: int) -> None:
        logger.info(f"Setting random seeds to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        set_seed(seed)

    @property
    def epochs(self) -> tqdm:
        return tqdm(
            range(self.epoch_idx, self.config.epochs),
            desc="Epochs",
            unit="epoch",
            disable=self.global_rank != 0,
        )

    @property
    def train_batches(self) -> tqdm:
        return tqdm(
            self.train_dataloader,
            desc="Train Batches",
            unit="batch",
            disable=self.global_rank != 0,
        )

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.global_rank}")

    @property
    def training_horizon(self) -> int:
        if self.train_dataloader is None:
            raise ValueError("Train dataloader not initialized.")
        if self.config.train_iters:
            return self.config.train_iters
        return (
            self.config.epochs
            * len(self.train_dataloader)
            // self.config.gradient_accumulation_steps
        )

    @property
    def warmup_steps(self) -> int:
        return int(self.config.scheduler.warmup_ratio * self.training_horizon)

    @callback_wrapper("loss")
    def loss(self, batch) -> Any:
        raise NotImplementedError("Loss method must be implemented by the trainer.")

    @callback_wrapper("step")
    def step(self, batch) -> None:
        self.global_step = self.model.global_steps
        self.model.train()
        loss = self.loss(batch)
        self.model.backward(loss)
        self.model.step()

    @callback_wrapper("epoch")
    def epoch(self) -> None:
        self.train_batch_idx = 0
        for batch in self.train_batches:
            self.train_batch_idx += 1
            self.global_step += 1
            self.step(batch)
            if self.early_stop:
                break

    @callback_wrapper("train")
    def train(self) -> None:
        try:
            for epoch_idx in self.epochs:
                self.epoch_idx = epoch_idx
                self.epoch()
                if self.early_stop:
                    break
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            # logger.info(f"{self._trainer_state}")
            raise (e)
