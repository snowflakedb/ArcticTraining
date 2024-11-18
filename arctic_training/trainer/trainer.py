import random
from abc import ABC
from abc import abstractmethod
from functools import wraps
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import deepspeed
import numpy as np
import torch
from deepspeed.accelerator import get_accelerator
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

try:
    from transformers.integrations.deepspeed import HfDeepSpeedConfig
except ImportError:
    from transformers.deepspeed import HfDeepSpeedConfig

from tqdm import tqdm
from transformers import set_seed

from arctic_training.callback.callback import Callback
from arctic_training.callback.factory import callback_factory
from arctic_training.checkpoint.checkpoint import CheckpointEngine
from arctic_training.checkpoint.factory import checkpoint_factory
from arctic_training.config.config import Config
from arctic_training.data.factory import data_factory
from arctic_training.logging import logger
from arctic_training.model.factory import model_factory
from arctic_training.optimizer.factory import optimizer_factory
from arctic_training.scheduler.factory import scheduler_factory


def _callback_wrapper(name: str) -> Callable:
    def decorator(method: Callable) -> Callable:
        @wraps(method)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            self._run_callbacks(f"pre-{name}")
            result = method(self, *args, **kwargs)
            self._run_callbacks(f"post-{name}")
            return result

        return wrapper

    return decorator


class TrainerState:
    _attributes: Dict[str, Any] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        super().__getattribute__("_attributes")[name] = value

    def __getattr__(self, name: str) -> Any:
        if name not in self._attributes:
            raise AttributeError(f"Attribute {name} not tracked in TrainerState.")
        return self._attributes.get(name)

    def __repr__(self) -> str:
        output = "TrainerState:\n"
        output += "\n".join([f"\t{k}: {v}" for k, v in self._attributes.items()])
        return output


class Trainer(ABC):
    config_type: Union[str, Type[Config]]
    dataset_type: str
    model_loader: Optional[Callable] = None
    training_finished: bool = False

    _default_callbacks: List[Tuple[str, Callable]] = []
    _trainer_callbacks: List[Tuple[str, Callable]] = []

    _trainer_state: TrainerState = TrainerState()

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        else:
            setattr(self._trainer_state, name, value)

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return super().__getattribute__(name)
        return getattr(self._trainer_state, name)

    def __init__(self, config: Config) -> None:
        logger.info(f"Initializing {self.__class__.__name__} trainer")

        self.config: Config
        self.train_dataloader: DataLoader
        self.eval_dataloader: Optional[DataLoader]
        self.model: PreTrainedModel
        self.optimizer: Any
        self.scheduler: Any
        self.callbacks: List[Callback]
        self.checkpoint_engines: List[CheckpointEngine]
        self._epoch_idx: int = 0
        self._global_step_idx: int = 0  # TODO: Use this value in checkpoint engine
        self._loss_output: Any = None

        self.config = config

        # Create callbacks first so we can call pre-init callbacks
        self._trainer_callbacks.extend(self._default_callbacks)
        self.callbacks = callback_factory(self)

        if self.local_rank == -1:
            self.device = torch.device(get_accelerator().device_name())
        else:
            get_accelerator().set_device(self.local_rank)
            self.device = torch.device(get_accelerator().device_name(), self.local_rank)
            deepspeed.init_distributed()

        self._run_callbacks("pre-init")
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        set_seed(self.config.seed)

        self.tokenizer, self.train_dataloader, self.eval_dataloader = data_factory(self)
        dschf = HfDeepSpeedConfig(self.config.deepspeed)  # noqa: F841
        self._run_callbacks("pre-model-init")
        self.model = model_factory(self)
        self._run_callbacks("post-model-init")
        self._run_callbacks("pre-optimizer-init")
        self.optimizer = optimizer_factory(self)
        self._run_callbacks("post-optimizer-init")
        self.scheduler = scheduler_factory(self)
        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            args=self.config,
            lr_scheduler=self.scheduler,
            config=self.config.deepspeed,
        )

        self.checkpoint_engines = self.checkpoint_engine()

        self._run_callbacks("post-init")
        if self.local_rank == 0:
            print("CONFIG")
            print(self.config)

    def checkpoint_engine(self):
        return checkpoint_factory(self)

    def _run_callbacks(self, event: str) -> None:
        for cb in self.callbacks:
            if cb.event == event:
                logger.info(f"Running callback: {cb} at event: {event}")
                cb(self)

    @property
    def local_rank(self) -> int:
        return self.config.local_rank

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
        return int(self.config.warmup_ratio * self.training_horizon)

    @abstractmethod
    def loss(self, data_batch) -> None:
        raise NotImplementedError(
            "Inheriting Trainer classes should define the loss method."
        )

    @_callback_wrapper("loss")
    def _loss(self, batch_data) -> None:
        self._loss_output = self.loss(batch_data)

    def step(self) -> None:
        self.model.train()
        self._loss(self.train_batch_data)
        self._save_checkpoint()  # TODO: must this happen before backward?
        self.model.backward(self._loss_output)
        self.model.step()

    @_callback_wrapper("step")
    def _step(self) -> None:
        self.step()

    def train_batch_loop(self) -> None:
        self._step()
        if self.local_rank == 0:
            logger.info(
                f"EPOCH: {self.epoch_idx}, BATCH: {self.train_batch_idx}, LOSS: {self._loss_output.item()}"
            )

    @_callback_wrapper("batch")
    def _train_batch_loop(self) -> None:
        self.train_batch_loop()

    @property
    def train_batches(self) -> tqdm:
        if self.local_rank == 0:
            logger.info(f"Total training batches: {len(self.train_dataloader)}")
        return tqdm(enumerate(self.train_dataloader), desc="Batches", unit="batch")

    def epoch_loop(self) -> None:
        for train_batch_idx, train_batch_data in self.train_batches:
            self.train_batch_idx = train_batch_idx
            self.train_batch_data = train_batch_data
            self._train_batch_loop()

    @_callback_wrapper("epoch")
    def _epoch_loop(self) -> None:
        self.epoch_loop()

    def eval_batch_loop(self) -> None:
        with torch.on_grad():
            self._loss(self.eval_batch_data)

    @_callback_wrapper("eval-batch")
    def _eval_batch_loop(self) -> None:
        self.eval_batch_loop()

    def validate(self) -> None:
        self.model.eval()
        for eval_batch_idx, eval_batch_data in self.eval_batches:
            self.eval_batch_idx = eval_batch_idx
            self.eval_batch_data = eval_batch_data
            self._eval_batch_loop()

    @_callback_wrapper("validate")
    def _validate(self) -> None:
        self.validate()

    @_callback_wrapper("checkpoint")
    def _save_checkpoint(self) -> None:
        for engine in self.checkpoint_engines:
            if engine.do_checkpoint:
                engine.save()

    @_callback_wrapper("resume_checkpoint")
    def _load_checkpoint(self) -> None:
        for engine in self.checkpoint_engines:
            if engine.load_checkpoint:
                engine.load()

    @_callback_wrapper("training")
    def train(self) -> None:
        try:
            for epoch_idx in self.epochs:
                self.epoch_idx = epoch_idx
                self._epoch_loop()
                if self.early_stop:
                    break
                self._save_checkpoint()
            self.training_finished = True
            self._save_checkpoint()
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            logger.info(f"{self._trainer_state}")
            raise e

    @property
    def epoch_idx(self) -> int:
        return self._epoch_idx

    @epoch_idx.setter
    def epoch_idx(self, value: int) -> None:
        # TODO: Use this to set end_of_training and early_stop?
        if value >= self.config.epochs:
            raise ValueError("Epoch index cannot be greater than total epochs.")
        self._epoch_idx = value

    @property
    def global_step_idx(self) -> int:
        return self._global_step_idx

    @global_step_idx.setter
    def global_step_idx(self, value: int) -> None:
        if value >= self.training_horizon:
            raise ValueError(
                "Global step index cannot be greater than total training steps."
            )
        self._global_step_idx = value

    @property
    def epochs(self) -> tqdm:
        return tqdm(
            range(self.epoch_idx, self.config.epochs), desc="Epochs", unit="epoch"
        )
