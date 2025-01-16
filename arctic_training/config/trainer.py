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

import importlib.util
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import yaml
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from arctic_training.checkpoint.engine import CheckpointEngine

import deepspeed
from deepspeed.accelerator import get_accelerator
from pydantic import ValidationInfo

from arctic_training.config import BaseConfig
from arctic_training.registry.checkpoint import get_registered_checkpoint_engine
from arctic_training.registry.data import get_registered_data_factory
from arctic_training.registry.model import get_registered_model_factory
from arctic_training.registry.optimizer import get_registered_optimizer_factory
from arctic_training.registry.scheduler import get_registered_scheduler_factory
from arctic_training.registry.tokenizer import get_registered_tokenizer_factory
from arctic_training.registry.trainer import get_registered_trainer
from arctic_training.utils import get_local_rank
from arctic_training.utils import get_world_size

from .checkpoint import CheckpointConfig
from .data import DataConfig
from .logger import LoggerConfig
from .model import ModelConfig
from .optimizer import OptimizerConfig
from .scheduler import SchedulerConfig
from .tokenizer import TokenizerConfig
from .wandb import WandBConfig

TRAINER_DEFAULT = "sft"
CUSTOM_CODE_DEFAULT = Path("train.py")


class TrainerConfig(BaseConfig):
    """Base Trainer Configuration."""

    type: str = TRAINER_DEFAULT
    """ Trainer type. """

    code: Path = CUSTOM_CODE_DEFAULT
    """ Path to the python script containing custom trainer implementation. """

    model: ModelConfig
    """ Model configuration. """

    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    """ Tokenizer configuration. """

    data: DataConfig
    """ Train and eval data configuration. """

    logger: LoggerConfig = Field(default_factory=LoggerConfig)
    """ Logger configuration. """

    wandb: WandBConfig = Field(default_factory=WandBConfig)
    """ Weights and Biases configuration. """

    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    """ Scheduler configuration. """

    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    """ Optimizer configuration. """

    deepspeed: Dict[str, Any] = {}
    """ DeepSpeed config dict. Will be automatically filled if not provided by the user. """

    epochs: int = Field(default=1, ge=0)
    """ Number of epochs to train. """

    loss_log_interval: int = Field(default=1, ge=0)
    """ Number of steps between logging loss. """

    gradient_accumulation_steps: int = 1
    """ Number of gradient accumulation steps. """

    micro_batch_size: int = Field(default=1, ge=1)
    """ Micro batch size per GPU. """

    seed: int = Field(default=42, ge=0)
    """ Random seed value for numpy, python.random, torch, and transformers. """

    checkpoint: List[CheckpointConfig] = []
    """ Checkpoint configurations. Multiple checkpoint engines may be used together. """

    train_iters: int = Field(default=0, ge=0)
    """ Maximum number of training iterations. """

    eval_frequency: int = Field(default=0, ge=0)

    global_rank: int = Field(default_factory=get_local_rank, exclude=True)
    world_size: int = Field(default_factory=get_world_size, exclude=True)

    exit_iteration: int = Field(default=0, ge=0)
    """ Force exit of training after specified iteration count (useful for debugging). """

    @model_validator(mode="after")
    def init_dist(self) -> Self:
        get_accelerator().set_device(self.global_rank)
        deepspeed.init_distributed()
        return self

    # TODO deprecate scheduler LR and move to optimizer LR
    @model_validator(mode="after")
    def copy_lr(self) -> Self:
        self.optimizer.learning_rate = self.scheduler.learning_rate
        return self

    @property
    def trainer(self):
        return get_registered_trainer(self.type)(config=self)

    @property
    def checkpoint_engines(self) -> List[partial["CheckpointEngine"]]:
        checkpoint_engines = []
        for checkpoint in self.checkpoint:
            checkpoint_engine = get_registered_checkpoint_engine(checkpoint.type)
            checkpoint_engines.append(partial(checkpoint_engine, config=checkpoint))
        return checkpoint_engines

    @property
    def zero_3_enabled(self) -> bool:
        return self.deepspeed.get("zero_optimization", {}).get("stage", 0) == 3

    @field_validator("eval_frequency", mode="after")
    def validate_eval_frequency(cls, v: int, info: ValidationInfo) -> int:
        if (
            info.data["data"].eval_sources
            or info.data["data"].train_eval_split[1] > 0.0
        ):
            assert v > 0, "eval_frequency must be set if eval dataset is provided."
        return v

    @field_validator("tokenizer", mode="after")
    def set_tokenizer(cls, v: TokenizerConfig, info: ValidationInfo) -> TokenizerConfig:
        if not v.name_or_path and "model" in info.data:
            v.name_or_path = info.data["model"].name_or_path
        return v

    @field_validator(
        "checkpoint",
        "data",
        "model",
        "optimizer",
        "scheduler",
        "tokenizer",
        mode="before",
    )
    @classmethod
    def parse_sub_config(
        cls,
        v: Any,
        info: ValidationInfo,
    ) -> Union[BaseConfig, List[BaseConfig]]:
        trainer_attr_map = {
            "checkpoint": "checkpoint_engine_type",
            "data": "data_factory_type",
            "model": "model_factory_type",
            "optimizer": "optimizer_factory_type",
            "scheduler": "scheduler_factory_type",
            "tokenizer": "tokenizer_factory_type",
        }
        field_name: str = info.field_name  # type: ignore
        trainer_type: str = info.data["type"]
        trainer_cls = get_registered_trainer(trainer_type)
        trainer_field_default = getattr(trainer_cls, trainer_attr_map[field_name])[0]

        if isinstance(v, tuple) or isinstance(v, list):
            return_list = []
            for sub_v in v:
                if isinstance(sub_v, BaseConfig):
                    sub_v = sub_v.model_dump()
                field_cls = cls._get_config_cls(
                    sub_v, field_name, trainer_field_default
                )
                sub_v["type"] = field_cls.name
                return_list.append(field_cls.config_type(**sub_v))
            return return_list

        if isinstance(v, BaseConfig):
            v = v.model_dump()
        field_cls = cls._get_config_cls(v, field_name, trainer_field_default)
        v["type"] = field_cls.name
        return field_cls.config_type(**v)

    @classmethod
    def _get_config_cls(cls, config_dict, field_name, default_cls):
        get_class_fn_map = {
            "checkpoint": get_registered_checkpoint_engine,
            "data": get_registered_data_factory,
            "model": get_registered_model_factory,
            "optimizer": get_registered_optimizer_factory,
            "scheduler": get_registered_scheduler_factory,
            "tokenizer": get_registered_tokenizer_factory,
        }
        field_type = config_dict.get("type", "")
        if field_type == "":
            field_type = default_cls
        field_cls = get_class_fn_map[field_name](field_type)
        return field_cls

    @field_validator("logger", mode="after")
    @classmethod
    def initialize_logger(cls, v: LoggerConfig) -> LoggerConfig:
        from arctic_training.logging import setup_logger

        setup_logger(v)
        return v

    @field_validator("checkpoint", mode="before")
    def checkpoint_to_list(cls, v: Union[Dict, List[Dict]]) -> List[Dict]:
        if not isinstance(v, list):
            return [v]
        return v

    @model_validator(mode="after")
    def build_deepspeed_config(self) -> Self:
        ds_config = self.deepspeed
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size
        ds_config["train_batch_size"] = (
            self.micro_batch_size * self.gradient_accumulation_steps * self.world_size
        )
        ds_config["steps_per_print"] = ds_config.get("steps_per_print", 10)
        ds_config["zero_optimization"] = ds_config.get(
            "zero_optimization",
            {
                "stage": 2,
                "stage3_param_persistence_threshold": 1e4,
                "stage3_max_live_parameters": 3e7,
                "stage3_prefetch_bucket_size": 3e7,
                "memory_efficient_linear": False,
            },
        )
        ds_config["bfloat16"] = ds_config.get("bfloat16", {"enabled": True})
        ds_config["gradient_clipping"] = ds_config.get("gradient_clipping", 1.0)
        ds_config["prescale_gradients"] = ds_config.get("prescale_gradients", False)
        ds_config["wall_clock_breakdown"] = ds_config.get("wall_clock_breakdown", False)
        return self

    @model_validator(mode="after")
    def validate_single_checkpoint_resume(self) -> Self:
        resume_checkpoint_values = [c.auto_resume for c in self.checkpoint]
        assert (
            sum(resume_checkpoint_values) <= 1
        ), "Only one checkpoint can auto resume."
        return self


def get_config(config_file: Path) -> BaseConfig:
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    config_dir = config_file.parent

    trainer_type = config_dict.get("type", TRAINER_DEFAULT)
    config_dict["type"] = trainer_type

    trainer_script = config_dict.get("code", CUSTOM_CODE_DEFAULT)
    config_dict["code"] = trainer_script

    script_path = Path(trainer_script)
    if not script_path.is_absolute():
        script_path = config_dir / script_path
    script_path = script_path.resolve()

    if script_path.exists():
        module_name = "custom_trainer"
        script_dir = str(script_path.parent)
        original_sys_path = sys.path.copy()
        sys.path.insert(0, script_dir)
        try:
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None:
                raise ImportError(f"Cannot load script from {script_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore
        finally:
            sys.path = original_sys_path

    trainer_cls = get_registered_trainer(trainer_type)
    config_cls = trainer_cls.config_type

    config = config_cls(**config_dict)

    return config
