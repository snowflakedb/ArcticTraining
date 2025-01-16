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
from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
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
    """Base Trainer class."""

    name: str
    """
    Name of the trainer used for registering custom trainers. This name
    should be unique and is used in the training recipe YAMLs to identify which
    trainer to be used.
    """

    config_type: Type[TrainerConfig]
    """
    The type of the config class that the trainer uses. This should be a
    subclass of TrainerConfig and add any trainer-specific fields.
    """

    data_factory_type: List[Type["DataFactory"]]
    """
    A List of valid data factory types that the trainer can use. These should
    inherit from DataFactory. The first item in the list will be used as the
    default if the type is not explicitly set in the YAML config.
    """

    model_factory_type: List[Type["ModelFactory"]]
    """
    A List of valid model factory types that the trainer can use. These should
    inherit from ModelFactory. The first item in the list will be used as the
    default if the type is not explicitly set in the YAML config.
    """

    checkpoint_engine_type: List[Type["CheckpointEngine"]]
    """
    A List of valid checkpoint engine types that the trainer can use. These
    should inherit from CheckpointEngine. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    optimizer_factory_type: List[Type["OptimizerFactory"]]
    """
    A List of valid optimizer factory types that the trainer can use. These
    should inherit from OptimizerFactory. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    scheduler_factory_type: List[Type["SchedulerFactory"]]
    """
    A List of valid scheduler factory types that the trainer can use. These
    should inherit from SchedulerFactory. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    tokenizer_factory_type: List[Type["TokenizerFactory"]]
    """
    A List of valid tokenizer factory types that the trainer can use. These
    should inherit from TokenizerFactory. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    callbacks: List[Tuple[str, Callable]] = [post_loss_log_cb]
    """
    A list of callbacks for the trainer. Callbacks are specified as tuples of a
    string indicating where the callback should be placed and a callable that
    implements the callback. Callback events for the trainer include `pre-` and
    `post-` for `init`, `train`, `epoch`, `step`, and `checkpoint`.
    """

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
        """Epochs iterator."""
        return tqdm(
            range(self.epoch_idx, self.config.epochs),
            desc="Epochs",
            unit="epoch",
            disable=self.global_rank != 0,
        )

    @property
    def train_batches(self) -> tqdm:
        """Training data iterator."""
        return tqdm(
            self.train_dataloader,
            desc="Train Batches",
            unit="batch",
            disable=self.global_rank != 0,
        )

    @property
    def device(self) -> torch.device:
        """Current device."""
        return torch.device(f"cuda:{self.global_rank}")

    @property
    def training_horizon(self) -> int:
        """Total number of training iterations."""
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
        """Number of warmup steps."""
        return int(self.config.scheduler.warmup_ratio * self.training_horizon)

    @callback_wrapper("loss")
    @abstractmethod
    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Loss function for the trainer. This method should be implemented by the
        inheriting trainer class.
        """
        raise NotImplementedError("Loss method must be implemented by the trainer.")

    @callback_wrapper("step")
    def step(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Step function for the trainer. Each batch of training data is passed to
        this method.
        """

        self.model.train()
        loss = self.loss(batch)
        self.model.backward(loss)
        self.model.step()

        # use deepspeed global step as golden truth
        self.global_step = self.model.global_steps
        if self.global_step >= self.training_horizon:
            self.early_stop = True

        self.checkpoint()

        if (
            self.config.exit_iteration > 0
            and self.config.exit_iteration == self.global_step
        ):
            logger.info(f"Hit exit iteration of {self.global_step}, forcing exit")
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
            exit()

    @callback_wrapper("epoch")
    def epoch(self) -> None:
        """
        Epoch training loop. This method will be called for each epoch of
        training and iterates across batches of training data, calling the step
        method on each batch.
        """
        self.train_batch_idx = 0
        for batch in self.train_batches:
            self.train_batch_idx += 1

            self.step(batch)
            if self.early_stop:
                break

    @callback_wrapper("train")
    def train(self) -> None:
        """
        Main training loop. Calls the epoch method for each epoch of training.
        """
        try:
            for epoch_idx in self.epochs:
                self.epoch_idx = epoch_idx
                self.epoch()
                if self.early_stop:
                    break
                self.checkpoint()
            self.training_finished = True
            logger.info("Training finished.")
            self.checkpoint()
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            # logger.info(f"{self._trainer_state}")
            raise (e)

    @callback_wrapper("checkpoint")
    def checkpoint(self) -> None:
        for engine in self.checkpoint_engines:
            if engine.do_checkpoint:
                logger.info(f"Saving Checkpoint at global step: {self.global_step}.")
                engine.save(self.model)
