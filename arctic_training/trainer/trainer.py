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


import math
import random
from abc import ABC
from abc import abstractmethod
from functools import cached_property
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import deepspeed
import numpy as np
import torch
import torch.cuda
import torch.distributed.nn
import wandb
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPAttentionHF
from deepspeed.runtime.sequence_parallel.ulysses_sp import UlyssesSPDataLoaderAdapter
from devtools import debug
from tqdm import tqdm
from transformers import set_seed
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from wandb.sdk.wandb_run import Run as WandbRun

from arctic_training.callback.logging import post_loss_log_cb
from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.checkpoint.engine import CheckpointEngine
from arctic_training.config.trainer import TrainerConfig
from arctic_training.data.factory import DataFactory
from arctic_training.data.utils import OverfitOneBatchDataLoader
from arctic_training.logging import logger
from arctic_training.metrics import Metrics
from arctic_training.model.factory import ModelFactory
from arctic_training.model.tiled_compute import enable_tiled_mlp_compute
from arctic_training.optimizer.factory import OptimizerFactory
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method
from arctic_training.scheduler.factory import SchedulerFactory
from arctic_training.tokenizer.factory import TokenizerFactory
from arctic_training.utils import append_json_file


class Trainer(ABC, CallbackMixin, metaclass=RegistryMeta):
    """Base Trainer class."""

    name: str
    """
    Name of the trainer used for registering custom trainers. This name
    should be unique and is used in the training recipe YAMLs to identify which
    trainer to be used.
    """

    config: TrainerConfig
    """
    The type of the config class that the trainer uses. This should be a
    subclass of TrainerConfig and add any trainer-specific fields.
    """

    data_factory: DataFactory
    """
    A List of valid data factory types that the trainer can use. These should
    inherit from DataFactory. The first item in the list will be used as the
    default if the type is not explicitly set in the YAML config.
    """

    model_factory: ModelFactory
    """
    A List of valid model factory types that the trainer can use. These should
    inherit from ModelFactory. The first item in the list will be used as the
    default if the type is not explicitly set in the YAML config.
    """

    checkpoint_engine: CheckpointEngine
    """
    A List of valid checkpoint engine types that the trainer can use. These
    should inherit from CheckpointEngine. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    optimizer_factory: OptimizerFactory
    """
    A List of valid optimizer factory types that the trainer can use. These
    should inherit from OptimizerFactory. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    scheduler_factory: SchedulerFactory
    """
    A List of valid scheduler factory types that the trainer can use. These
    should inherit from SchedulerFactory. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    tokenizer_factory: TokenizerFactory
    """
    A List of valid tokenizer factory types that the trainer can use. These
    should inherit from TokenizerFactory. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    callbacks: List[Tuple[str, Callable]] = [
        post_loss_log_cb,
    ]
    """
    A list of callbacks for the trainer. Callbacks are specified as tuples of a
    string indicating where the callback should be placed and a callable that
    implements the callback. Callback events for the trainer include `pre-` and
    `post-` for `init`, `train`, `epoch`, `step`, and `checkpoint`.
    """

    @classmethod
    def _validate_subclass(cls) -> None:
        _validate_class_attribute_set(cls, "name")
        _validate_class_attribute_type(cls, "config", TrainerConfig)
        _validate_class_attribute_type(cls, "data_factory", DataFactory)
        _validate_class_attribute_type(cls, "model_factory", ModelFactory)
        _validate_class_attribute_type(cls, "checkpoint_engine", CheckpointEngine)
        _validate_class_attribute_type(cls, "optimizer_factory", OptimizerFactory)
        _validate_class_attribute_type(cls, "scheduler_factory", SchedulerFactory)
        _validate_class_attribute_type(cls, "tokenizer_factory", TokenizerFactory)
        _validate_class_method(cls, "loss", ["self", "batch"])
        _validate_class_method(cls, "step", ["self", "batch"])
        _validate_class_method(cls, "epoch", ["self"])
        _validate_class_method(cls, "train", ["self"])
        _validate_class_method(cls, "checkpoint", ["self"])

    def __init__(self, config: TrainerConfig, mode: str = "train") -> None:
        logger.info(f"Initializing Trainer with config:\n{debug.format(config)}")
        self.config = config
        self.epoch_idx = 0
        self.train_batch_idx = 0
        self.global_step = 0
        self.eval_batch_idx = 0
        self.early_stop = False
        self.world_size = config.world_size
        self.global_rank = config.global_rank
        self.epoch_finished = False
        self.training_finished = False
        self.wandb_experiment: Optional[WandbRun] = None

        self._set_seeds(self.config.seed)

        if self.config.mem_profiler == "e2e":
            torch.cuda.memory._record_memory_history(max_entries=self.config.mem_profiler_max_entries)

        tokenizer_factory = self.config.tokenizer.factory(self)
        self.tokenizer = tokenizer_factory()

        data_factory = self.config.data.factory(self)
        self.train_dataloader, self.eval_dataloader_map = data_factory()
        if mode == "process-data":
            return

        if self.config.overfit_first_batch:
            self.train_dataloader = OverfitOneBatchDataLoader(self.train_dataloader)

        # XXX: We can abstract this section further with AT-specific wrapper, but
        # UlyssesSPAttentionHF should not have any AT-specific objects / assumptions
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=self.config.model.name_or_path,
            core_attn_implementation=self.config.model.attn_implementation,
            sequence_parallel_size=self.config.sequence_parallel_size,
            max_length=self.config.data.max_length,
            micro_batch_size=self.config.micro_batch_size,
            seq_length_is_variable=True,
        )

        # Important: this is most likely not beneficial under seqlen=64k
        if self.config.activation_checkpoint_cpu_offload:
            # activation_checkpointing_cpu_offload becomes very benefitial at very long seqlen
            # e.g., llama 8b at 800k (100k effective per gpu) will save 24GB per gpu:
            # ((100_000*4096)*2*32/2**30), but for short sequences the offload will just slow things
            # down,
            #
            # XXX: could parameterize or run a few lengths to see at which threshold it becomes
            # beneficial - a user might still want this on even at shorter seqlen if they don't
            # mind slower performance. discussing adding this functionality to pytorch core
            # (https://pytorch.slack.com/archives/C3PDTEV8E/p1745274102600729)
            from arctic_training.monkey_patches import monkey_patch_checkpoint_function_with_cpu_offload

            monkey_patch_checkpoint_function_with_cpu_offload()

        # MLP tiling - has to happen before model is instantiated
        if self.config.tiled_mlp_compute:
            enable_tiled_mlp_compute(self.config.model.name_or_path)

        dschf = HfDeepSpeedConfig(self.config.deepspeed)  # noqa: F841
        model_factory = self.config.model.factory(self)
        self.model = model_factory()

        # XXX: not sure when to enable this temp hack until HF Transformers comes up with a way to
        # override this properly - apparently there is a plan to do so in the future versions of
        # transformers.
        # 1. definitely needed for SP
        # 2. but it also should benefit a single gpu use case
        if self.config.sequence_parallel_size > 1 and self.config.model.attn_implementation != "flash_attention_2":
            # prevent from causal mask being created in HF Transformers - it's a huge `[bs, seqlen, seqlen]` tensor
            model_without_head = self.model_unwrapped.model
            if hasattr(model_without_head, "_update_causal_mask"):
                model_without_head._update_causal_mask = lambda *args: None

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
            mpu=mpu,
        )

        if self.config.sequence_parallel_size > 1:
            # deepspeed.initialize needs to run first
            from deepspeed.utils import groups

            # set SP-trainer attributes to be used later
            self.sp_group = groups._get_sequence_parallel_group()
            self.sp_world_size = groups._get_sequence_parallel_world_size()
            self.sp_rank = groups._get_sequence_parallel_rank()

            # wrap the DL with Ulysses one
            self.train_dataloader = UlyssesSPDataLoaderAdapter(
                self.train_dataloader,
                sp_rank=self.sp_rank,
                sp_group=self.sp_group,
                sp_world_size=self.sp_world_size,
                device=self.device,
            )

        self.checkpoint_engines = [engine(self) for engine in self.config.checkpoint_engines]

        for engine in self.checkpoint_engines:
            if engine.config.auto_resume:
                engine.load(self.model)

        self.metrics = Metrics(self)

        if self.global_rank == 0 and self.config.wandb.enable:
            # Note: wandb.init() is not type annotated so we need to use type: ignore
            self.wandb_experiment = wandb.init(  # type: ignore
                entity=self.config.wandb.entity,
                project=self.config.wandb.project,
                name=self.config.wandb.name,
                config=self.config.model_dump(),
            )

    def _set_seeds(self, seed: int) -> None:
        logger.info(f"Setting random seeds to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        set_seed(seed)

    @property
    def model_unwrapped(self):
        """Return the original model before it was wrapped by deepspeed"""
        if hasattr(self.model, "module"):
            return self.model.module
        else:
            return self.model

    @property
    def epochs(self) -> tqdm:
        """Epochs iterator."""
        total_epochs = self.config.epochs
        if self.config.train_iters:
            total_epochs = math.ceil(
                self.config.train_iters * self.config.gradient_accumulation_steps / len(self.train_dataloader)
            )

        return tqdm(
            range(self.epoch_idx, total_epochs),
            desc="Epochs",
            unit="epoch",
            disable=(self.global_rank != 0) or (self.config.train_log_iter_interval != 0),
        )

    @property
    def train_batches(self) -> tqdm:
        """Training data iterator."""
        return tqdm(
            self.train_dataloader,
            desc="Train Batches",
            unit="batch",
            disable=(self.global_rank != 0) or (self.config.train_log_iter_interval != 0),
        )

    @cached_property
    def device(self) -> torch.device:
        """Current device."""
        return torch.device(get_accelerator().device_name(self.config.local_rank))

    @property
    def training_horizon(self) -> int:
        """Total number of training iterations."""
        if self.train_dataloader is None:
            raise ValueError("Train dataloader not initialized.")
        if self.config.train_iters:
            return self.config.train_iters

        # XXX: this was incorrect for GAS
        return self.config.epochs * len(self.train_dataloader)  # // self.config.gradient_accumulation_steps

    @callback_wrapper("loss")
    @abstractmethod
    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Loss function for the trainer. This method should be implemented by the
        inheriting trainer class.
        """
        raise NotImplementedError("Loss method must be implemented by the trainer.")

    @callback_wrapper("backward")
    def backward(self, loss: torch.Tensor) -> None:
        """
        Backward function for the trainer. This method is called after the loss
        method and is responsible for backpropagating the loss through the model.
        """
        self.model.backward(loss)

    @callback_wrapper("step")
    def step(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Step function for the trainer. Each batch of training data is passed to
        this method.
        """

        self.model.train()

        loss = self.loss(batch)

        self.backward(loss)

        def maybe_item(v):
            return v.item() if torch.is_tensor(v) else v

        self.metrics.record("loss", maybe_item(loss))

        self.model.step()

        # use deepspeed global step as golden truth
        self.global_step = self.model.global_steps
        if self.global_step >= self.training_horizon:
            self.early_stop = True

        self.checkpoint()

        if self.config.exit_iteration > 0 and self.config.exit_iteration == self.global_step:
            self.early_stop = True
            logger.info(f"Hit exit iteration of {self.global_step}, ending training")

    @callback_wrapper("epoch")
    def epoch(self) -> None:
        """
        Epoch training loop. This method will be called for each epoch of
        training and iterates across batches of training data, calling the step
        method on each batch.
        """
        self.epoch_finished = False
        self.metrics.start_timer("iter")

        # enable memory allocation history, which will add tracebacks and event history to memory snapshots
        if self.config.mem_profiler == "step":
            torch.cuda.memory._record_memory_history(max_entries=self.config.mem_profiler_max_entries)

        for batch in self.train_batches:
            self.train_batch_idx += 1

            if (
                self.config.gradient_accumulation_steps == 1
                or self.train_batch_idx % self.config.gradient_accumulation_steps == 0
            ):
                self.gas_boundary = True
            else:
                self.gas_boundary = False

            if "packed_sample_seqlens" in batch and self.config.model.attn_implementation == "flash_attention_2":
                # deal correctly with packed samples under FA2, by calculating each seqlen tflos separately
                sample_seqlens = batch.pop("packed_sample_seqlens")
            else:
                sample_seqlens = [
                    [len(batch["input_ids"][idx]) * self.config.sequence_parallel_size]
                    for idx in range(len(batch["input_ids"]))
                ]
            self.metrics.seqlens = sample_seqlens

            self.metrics.start_timer("step")
            self.step(batch)
            self.metrics.stop_timer("step")

            self.metrics.restart_timer("iter")

            if (
                self.config.train_log_iter_interval != 0
                and self.train_batch_idx % self.config.train_log_iter_interval == 0
            ):
                self.metrics.print_summary()
                if self.global_rank == 0 and self.gas_boundary:

                    metrics = {k: v for k, v in self.metrics.summary_dict.items()}

                    append_json_file(self.config.train_log_metrics_path, metrics)

                    # first iter is a massive outlier for many fields - so skip it in wandb
                    if self.wandb_experiment is not None and self.train_batch_idx > 1:
                        metrics.pop("iter")  # not needed for wandb
                        self.wandb_experiment.log(metrics, step=self.model.global_steps)

            if self.config.kill_switch_path.exists():
                self.early_stop = True

            if self.early_stop:
                break
        self.metrics.stop_timer("iter")
        self.epoch_finished = True

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
        finally:
            if self.config.mem_profiler is not None:
                torch.cuda.memory._dump_snapshot(self.config.mem_profiler_dir / f"{self.global_rank}.pickle")

            if self.wandb_experiment is not None:
                self.wandb_experiment.finish()

    @callback_wrapper("checkpoint")
    def checkpoint(self) -> None:
        for engine in self.checkpoint_engines:
            if engine.do_checkpoint:
                logger.info(f"Saving Checkpoint at global step: {self.global_step}.")
                engine.save(self.model)
