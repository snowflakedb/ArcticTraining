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
from functools import cached_property
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import deepspeed
import numpy as np
import torch
import torch.distributed.nn
from deepspeed.accelerator import get_accelerator
from devtools import debug
from tqdm import tqdm
from transformers import set_seed
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from wandb.sdk.wandb_run import Run as WandbRun

import wandb
from arctic_training.callback.logging import post_loss_log_cb
from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.checkpoint.engine import CheckpointEngine
from arctic_training.config.trainer import TrainerConfig
from arctic_training.data.factory import DataFactory
from arctic_training.debug import print_rank
from arctic_training.debug import see_memory_usage
from arctic_training.logging import logger
from arctic_training.metrics import Metrics
from arctic_training.model.factory import ModelFactory
from arctic_training.optimizer.factory import OptimizerFactory
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method
from arctic_training.scheduler.factory import SchedulerFactory
from arctic_training.tokenizer.factory import TokenizerFactory

# XXX: this will be moved to deepspeed
if 1:
    from arctic_training.deepspeed import UlyssesSPAttentionHF
    from arctic_training.deepspeed import UlyssesSPDataLoaderWrapper


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

    # XXX: hack to compare correctness until we support GAS
    temp_losses = []

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

    def __init__(self, config: TrainerConfig) -> None:

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

        # enable memory history, which will add tracebacks and event history to snapshots
        # "none" | "e2e" | "step"
        self.mem_profiler = "none"
        # self.mem_profiler = "step"
        # profiling from here is slower, best to start at top of `epoch` ("step")
        if self.mem_profiler == "e2e":
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        # see_memory_usage("before model creation", force=True)

        tokenizer_factory = self.config.tokenizer.factory(self)
        self.tokenizer = tokenizer_factory()

        # see_memory_usage("after tokenizer", force=True)

        # dist.barrier()
        # see_memory_usage("before dataloader", force=True)

        data_factory = self.config.data.factory(self)
        self.train_dataloader, self.eval_dataloader = data_factory()

        # see_memory_usage("after dataloader", force=True)
        # exit()
        # XXX: eventually switch back to normal hf modeling code (it's just debug prints mod'ed at the moment)
        # there are no functional code changes in LlamaAttentionNew

        # transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttentionNew
        # XXX: We can abstract this section further with AT-specific wrapper, but UlyssesSPAttentionHF should not have any AT-specific objects / assumptions
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=self.config.model.name_or_path,
            core_attn_implementation=self.config.model.attn_implementation,
            sequence_parallel_size=self.config.sequence_parallel_size,
            max_length=self.config.data.max_length,
            micro_batch_size=self.config.micro_batch_size,
            seq_length_is_variable=True,
        )
        if self.config.sequence_parallel_size > 1:
            # we are overriding the original core attn implementation with `ulysses` and we have already passed the original core attn implementation to `UlyssesSPAttentionHF`
            self.config.model.attn_implementation = "ulysses"

        # see_memory_usage("after ulysses", force=True)

        dschf = HfDeepSpeedConfig(self.config.deepspeed)  # noqa: F841
        # print(self.config.deepspeed)
        model_factory = self.config.model.factory(self)
        self.model = model_factory()

        see_memory_usage("after model", force=True)

        UlyssesSPAttentionHF.validate_model(
            model=self.model,
            sequence_parallel_size=self.config.sequence_parallel_size,
        )

        optimizer_factory = self.config.optimizer.factory(self)
        self.optimizer = optimizer_factory()

        scheduler_factory = self.config.scheduler.factory(self)
        self.scheduler = scheduler_factory()

        torch.distributed.barrier()
        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            args=self.config,
            lr_scheduler=self.scheduler,
            config=self.config.deepspeed,
            mpu=mpu,
        )

        see_memory_usage("after ds", force=True)

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
        return tqdm(
            range(self.epoch_idx, self.config.epochs),
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
        return self.config.epochs * len(self.train_dataloader) // self.config.gradient_accumulation_steps

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

        # import deepspeed.comm as dist
        # import q
        # from deepspeed.utils import groups
        # q(self.global_rank)
        # print_rank0(f"{groups._get_sequence_parallel_group()=}")
        # print_rank0(f"{groups._get_sequence_parallel_rank()=}")
        # print_rank0(f"{groups._get_sequence_parallel_world_size()=}")
        # dist.barrier()
        # import time
        # time.sleep(5)
        # die

        torch.set_printoptions(sci_mode=False)
        # torch.set_printoptions(
        #     threshold=100000000, # print all data (without ... skipping) - can be huge!
        #     sci_mode=False,      # print all data on the same scale of 1 (this disables scientific notation)
        #     precision=6,         # print X decimal points for floats (default 4)
        #     edgeitems=5,         # when the data is large and skipped, control how many entries are printed on each edge
        #     linewidth=120,       # redefine linewidth for when lines are \n-wrapped in printout (default 80)
        #                         # if threshold is defined, matrix printing will ignore this setting
        #     profile="full",      # printing defaults: "default", "short", "full"
        # )

        # if self.global_rank == 0:
        #     print_rank0(batch)

        see_memory_usage("before forward", force=False)

        self.model.train()
        loss = self.loss(batch)
        self.model.backward(loss)

        # XXX: uncomment to compare loss exactness vs dp8-sp1
        # self.temp_losses.append(loss.item())
        # sp_world_size = 8
        # if len(self.temp_losses) == sp_world_size:
        #     avg_loss = sum(self.temp_losses) / len(self.temp_losses)
        #     print(f"{avg_loss=}")
        #     self.temp_losses = []

        # if self.config.sequence_parallel_size == 1:
        #     loss = self.loss(batch)
        #     self.model.backward(loss)

        #     # with torch.no_grad():
        #     #     # average losses since they are different on each dp rank
        #     #     losses_per_rank = torch.distributed.nn.functional.all_gather(loss)
        #     #     #print(f"LOSS {losses_per_rank=}")
        #     #     average_loss = torch.cat([l.unsqueeze(0) for l in losses_per_rank], dim=0).mean()
        #     #     #print(f"LOSS {average_loss=}")
        #     #     loss = average_loss

        # else:
        #     # sp will do backward inside sp_fwd_bwd_loss
        #     # the returned loss is already averaged across ranks and it's a float
        #     loss = self.sp_fwd_loss_bwd(batch)

        see_memory_usage("after backward", force=False)

        def maybe_item(v):
            return v.item() if torch.is_tensor(v) else v

        self.metrics.record("loss", maybe_item(loss))

        self.model.step()

        see_memory_usage("after step", force=False)
        # exit()

        # # should loss be averaged over sp sub-steps and logged as such?
        # loss = loss_aggregate / sp_world_size
        # print_rank0(f"averaged loss = {loss}")
        # #exit()

        # from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param

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

        see_memory_usage(f"entered epoch", force=True)
        # exit()

        # enable memory history, which will add tracebacks and event history to snapshots
        if self.mem_profiler == "step":
            torch.cuda.memory._record_memory_history(max_entries=100_000)

        train_batches = self.train_batches
        if self.config.sequence_parallel_size > 1:
            from deepspeed.utils import groups

            self.sp_group = groups._get_sequence_parallel_group()
            self.sp_world_size = groups._get_sequence_parallel_world_size()
            self.sp_rank = groups._get_sequence_parallel_rank()

            train_batches = UlyssesSPDataLoaderWrapper(
                train_batches,
                sp_rank=self.sp_rank,
                sp_group=self.sp_group,
                sp_world_size=self.sp_world_size,
                device=self.device,
            )
            # this will break on epoch 2+ as it'd continue multiplying the previous value from epoch 1
            self.config.exit_iteration *= self.sp_world_size
            # self.training_horizon *= self.sp_world_size
            self.metrics.max_iter *= self.sp_world_size

        # XXX: this counter must not be reset between epochs
        self.train_batch_idx = 0
        for batch in train_batches:
            self.train_batch_idx += 1
            print_rank(f"\n\n\n\n\nITERATION: {self.train_batch_idx} ", skip=False)

            self.metrics.record("seqlen", len(batch["input_ids"][0]) * self.config.sequence_parallel_size)

            see_memory_usage(f"before step", force=True)

            self.metrics.start_timer("step")
            self.step(batch)
            self.metrics.stop_timer("step")

            see_memory_usage(f"after step", force=True)

            self.metrics.restart_timer("iter")

            if (
                self.config.train_log_iter_interval != 0
                and self.train_batch_idx % self.config.train_log_iter_interval == 0
            ):
                self.metrics.print_summary()
                if (
                    self.global_rank == 0
                    and self.train_batch_idx > 1  # first iter is a massive outlier
                    and self.wandb_experiment is not None
                ):
                    self.wandb_experiment.log(
                        {k: v for k, v in self.metrics.summary_dict.items() if k != "iter"},
                        step=self.model.global_steps,
                    )

            if self.early_stop:
                break
        self.metrics.stop_timer("iter")
        self.epoch_finished = True

    @callback_wrapper("train")
    def train(self) -> None:
        """
        Main training loop. Calls the epoch method for each epoch of training.
        """

        # self.step_flos_counter = StepFlopCounter(start_iter=2)

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
            if self.mem_profiler == "e2e" or self.mem_profiler == "step":
                torch.cuda.memory._dump_snapshot(f"mem/mem_snapshot.{self.global_rank}.pickle")

            if self.wandb_experiment is not None:
                self.wandb_experiment.finish()

    @callback_wrapper("checkpoint")
    def checkpoint(self) -> None:
        for engine in self.checkpoint_engines:
            if engine.do_checkpoint:
                logger.info(f"Saving Checkpoint at global step: {self.global_step}.")
                engine.save(self.model)
