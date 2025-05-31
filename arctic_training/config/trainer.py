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
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Union
from typing import cast

import yaml
from pydantic import Field
from pydantic import ValidationInfo
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

from arctic_training.config.base import BaseConfig
from arctic_training.config.checkpoint import CheckpointConfig
from arctic_training.config.data import DataConfig
from arctic_training.config.enums import DType
from arctic_training.config.logger import LoggerConfig
from arctic_training.config.model import ModelConfig
from arctic_training.config.optimizer import OptimizerConfig
from arctic_training.config.scheduler import SchedulerConfig
from arctic_training.config.tokenizer import TokenizerConfig
from arctic_training.config.utils import HumanInt
from arctic_training.config.utils import UniqueKeyLoader
from arctic_training.config.utils import parse_human_val
from arctic_training.config.wandb import WandBConfig
from arctic_training.registry import _get_class_attr_type_hints
from arctic_training.registry import get_registered_checkpoint_engine
from arctic_training.registry import get_registered_data_factory
from arctic_training.registry import get_registered_model_factory
from arctic_training.registry import get_registered_optimizer_factory
from arctic_training.registry import get_registered_scheduler_factory
from arctic_training.registry import get_registered_tokenizer_factory
from arctic_training.registry import get_registered_trainer

if TYPE_CHECKING:
    from arctic_training.checkpoint.engine import CheckpointEngine

TRAINER_DEFAULT = "sft"
CUSTOM_CODE_DEFAULT = Path("train.py")


class TrainerConfig(BaseConfig):
    """Base Trainer Configuration."""

    type: str = TRAINER_DEFAULT
    """ Trainer type. """

    code: Path = CUSTOM_CODE_DEFAULT
    """ Path to the python script containing custom trainer implementation. """

    skip_validation: bool = False
    """ Skips validation of types for subconfigs and registered classes. """

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

    loss_log_interval: HumanInt = Field(default=1, ge=0)
    """ Number of steps between logging loss. """

    train_log_iter_interval: Literal[0, 1] = 1
    """ Iters between training metric log outputs. `0` is off, only intervals of `1` currently supported. """

    # XXX: fixme: the default output dir is broken
    # train_log_metrics_path: Path = Field(
    #     default_factory=lambda data: data["logger"].output_dir / "train-log-metrics.jsonl"
    # )
    # """ .jsonl path to log precise metrics according to the `train_log_iter_interval` schedule. Defaults to `logger.output_dir/train-log-metrics.jsonl` """
    train_log_metrics_path: Path = Path("train-log-metrics.jsonl")
    """ .jsonl path to log precise metrics according to the `train_log_iter_interval` schedule. Defaults to `./train-log-metrics.jsonl` """

    gradient_accumulation_steps: int = Field(default=1, ge=1)
    """ Number of gradient accumulation steps. """

    micro_batch_size: int = Field(default=1, ge=1)
    """ Micro batch size per GPU. """

    sequence_parallel_size: int = Field(default=1, ge=1)
    """ Sequence Parallelism Degree. Disabled if set to 1 """

    activation_checkpoint_cpu_offload: bool = False
    """ Offload activation checkpoint tensors to cpu. Enables a much longer sequence length. It is not very beneficial if sequence length is <64k  """

    tiled_mlp_compute: bool = False
    """ Tile the MLP computation to save GPU memory. Currently only limited architectures supported, but can be expanded to more. """

    seed: int = Field(default=42, ge=0)
    """ Random seed value for numpy, python.random, torch, and transformers. """

    checkpoint: List[CheckpointConfig] = []
    """ Checkpoint configurations. Multiple checkpoint engines may be used together. """

    train_iters: HumanInt = Field(default=0, ge=0)
    """ Maximum number of training iterations. """

    eval_frequency: int = Field(default=0, ge=0)

    exit_iteration: int = Field(default=0, ge=0)
    """ Force exit of training after specified iteration count (useful for debugging). """

    min_iterations: HumanInt = Field(default=0, ge=0)
    """ When >0, the training dataset will be replicated until there is enough data to run this many iterations. """

    overfit_first_batch: bool = False
    """ Train only on repetitions of the first training batch. Useful for development. """

    mem_profiler: Literal[None, "step", "e2e"] = None
    """ Enable memory profiling. """

    mem_profiler_dir: Path = Field(default_factory=lambda data: data["logger"].output_dir / "mem-prof")
    """ Path to save memory profiling results. Defaults to `logger.output_dir/mem-prof`. """

    mem_profiler_max_entries: HumanInt = Field(default=100_000, ge=1)
    """ Maximum number of entries to store in the memory profiler. """

    kill_switch_path: Path = Path("/tmp/at_kill_switch")
    """ Path to a file that can be used to trigger a graceful shutdown mid-training (sets early exit to True). """

    @model_validator(mode="after")
    def init_dist(self) -> Self:
        import deepspeed
        from deepspeed.accelerator import get_accelerator

        get_accelerator().set_device(self.local_rank)
        deepspeed.init_distributed()
        return self

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

    @staticmethod
    def _get_subconfig_object(
        v: Union[Dict, BaseConfig],
        info: ValidationInfo,
        get_class_fn: Callable,
        attr_name: str,
    ) -> BaseConfig:
        # Get the trainer class as it will tell us which types of factory
        # classes (and thus configs) are default/compatible
        trainer_type = info.data["type"]
        trainer_cls = get_registered_trainer(trainer_type)

        # Get type hints for this factory class. This is a list of compatible
        # classes for the given attribute field.
        attribute_type_hints = _get_class_attr_type_hints(trainer_cls, attr_name)

        # Convert to a dictionary as default values are the base config classes
        # and we likely need to use a different class based on the trainer type
        # or user requested `type` field value.
        if isinstance(v, dict):
            config_dict = v
        else:
            # Must exclude computed fields to avoid validation errors
            config_dict = v.model_dump(exclude={"local_rank", "global_rank", "world_size"})

        # Determine which attribute class to use (e.g., for `model`:
        # HFModelFactory, LigerModelFactory, etc.)
        if config_dict.get("type", ""):
            # User explicitly specified the type
            attr_cls = get_class_fn(config_dict["type"])
        else:
            # User did not specify the type, use the first (maybe only) hint as default type
            attr_cls = attribute_type_hints[0]

        # Check that the requested/resolved type is compatible with the trainer
        if not info.data.get("skip_validation") and attr_cls not in attribute_type_hints:
            raise ValueError(
                f"{attr_cls.__name__} is not supported for {attr_name} in"
                f" {trainer_cls.__name__}. Supported types are"
                f" {[cls.__name__ for cls in attribute_type_hints]}."
            )

        # Make sure the `type` field is set in the config dict
        config_dict["type"] = attr_cls.name

        # Get the config class for the factory class and creat the config
        config_cls = _get_class_attr_type_hints(attr_cls, "config")[0]
        return config_cls(**config_dict)

    @staticmethod
    def _to_list(v: Union[Any, List[Any]]) -> List[Any]:
        if not isinstance(v, list):
            return [v]
        return v

    @field_validator("checkpoint", mode="before")
    @classmethod
    def init_checkpoint_configs(
        cls,
        v: Union[Union[Dict, CheckpointConfig], List[Union[Dict, CheckpointConfig]]],
        info: ValidationInfo,
    ) -> List[CheckpointConfig]:
        v = cls._to_list(v)
        return_list = []
        for sub_v in v:
            return_list.append(
                cls._get_subconfig_object(
                    v=sub_v,
                    info=info,
                    get_class_fn=get_registered_checkpoint_engine,
                    attr_name="checkpoint_engine",
                )
            )
        return [cast(CheckpointConfig, subconfig) for subconfig in return_list]

    @field_validator("data", mode="before")
    @classmethod
    def init_data_config(cls, v: Union[Dict, DataConfig], info: ValidationInfo) -> DataConfig:
        subconfig = cls._get_subconfig_object(
            v=v,
            info=info,
            get_class_fn=get_registered_data_factory,
            attr_name="data_factory",
        )
        return cast(DataConfig, subconfig)

    @field_validator("model", mode="before")
    @classmethod
    def init_model_config(cls, v: Union[Dict, ModelConfig], info: ValidationInfo) -> ModelConfig:
        subconfig = cls._get_subconfig_object(
            v=v,
            info=info,
            get_class_fn=get_registered_model_factory,
            attr_name="model_factory",
        )
        return cast(ModelConfig, subconfig)

    @field_validator("optimizer", mode="before")
    @classmethod
    def init_optimizer_config(cls, v: Union[Dict, OptimizerConfig], info: ValidationInfo) -> OptimizerConfig:
        subconfig = cls._get_subconfig_object(
            v=v,
            info=info,
            get_class_fn=get_registered_optimizer_factory,
            attr_name="optimizer_factory",
        )
        return cast(OptimizerConfig, subconfig)

    @field_validator("scheduler", mode="before")
    @classmethod
    def init_scheduler_config(cls, v: Union[Dict, SchedulerConfig], info: ValidationInfo) -> SchedulerConfig:
        subconfig = cls._get_subconfig_object(
            v=v,
            info=info,
            get_class_fn=get_registered_scheduler_factory,
            attr_name="scheduler_factory",
        )
        return cast(SchedulerConfig, subconfig)

    @field_validator("tokenizer", mode="before")
    @classmethod
    def init_tokenizer_config(cls, v: Union[Dict, TokenizerConfig], info: ValidationInfo) -> TokenizerConfig:
        subconfig = cls._get_subconfig_object(
            v=v,
            info=info,
            get_class_fn=get_registered_tokenizer_factory,
            attr_name="tokenizer_factory",
        )
        return cast(TokenizerConfig, subconfig)

    @model_validator(mode="after")
    def validate_eval_frequency(self) -> Self:
        if self.data.eval_sources or self.data.train_eval_split[1] > 0.0:
            assert self.eval_frequency > 0, "eval_frequency must be set if eval dataset is provided."
        return self

    @model_validator(mode="after")
    def set_tokenizer(self) -> Self:
        if not self.tokenizer.name_or_path:
            self.tokenizer.name_or_path = self.model.name_or_path
        return self

    @field_validator("logger", mode="after")
    @classmethod
    def initialize_logger(cls, v: LoggerConfig) -> LoggerConfig:
        from arctic_training.logging import setup_logger

        setup_logger(v)
        return v

    @field_validator("deepspeed", mode="before")
    @classmethod
    def coerce_deepspeed_human_friendly_values(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        # Allow human friendly values for deepspeed config. This is a workaround
        # until we upstream this feature to the DeepSpeed pydantic configs.
        def coerce_dict_values(config_dict: Dict[str, Any]) -> Dict[str, Any]:
            coerced_dict: Dict[str, Any] = {}
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    coerced_dict[key] = coerce_dict_values(value)
                else:
                    try:
                        coerced_dict[key] = parse_human_val(value)
                    except Exception:
                        coerced_dict[key] = value
            return coerced_dict

        return coerce_dict_values(v)

    @model_validator(mode="after")
    def build_deepspeed_config(self) -> Self:
        ds_config = self.deepspeed
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_batch_size
        ds_config["train_batch_size"] = (
            self.micro_batch_size * self.gradient_accumulation_steps * self.world_size / self.sequence_parallel_size
        )
        ds_config["gradient_accumulation_steps"] = self.gradient_accumulation_steps
        ds_config["sequence_parallel_size"] = self.sequence_parallel_size
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
        if "bfloat16" not in ds_config:
            if self.model.dtype == DType.BF16:
                ds_config["bfloat16"] = {"enabled": True}
        if "fp16" not in ds_config:
            if self.model.dtype == DType.FP16:
                ds_config["fp16"] = {"enabled": True}
        ds_config["gradient_clipping"] = ds_config.get("gradient_clipping", 1.0)
        ds_config["prescale_gradients"] = ds_config.get("prescale_gradients", False)
        ds_config["wall_clock_breakdown"] = ds_config.get("wall_clock_breakdown", False)
        return self

    @model_validator(mode="after")
    def validate_single_checkpoint_resume(self) -> Self:
        resume_checkpoint_values = [c.auto_resume for c in self.checkpoint]
        assert sum(resume_checkpoint_values) <= 1, "Only one checkpoint can auto resume."
        return self

    @model_validator(mode="after")
    def train_log_metrics_path_prep(self) -> Self:
        if self.local_rank == 0:
            self.train_log_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.train_log_metrics_path.open(mode="a")
        return self

    @model_validator(mode="after")
    def mem_profiler_mkdir(self) -> Self:
        if self.mem_profiler is not None:
            self.mem_profiler_dir.mkdir(parents=True, exist_ok=True)
        return self


def get_config(config_file_or_dict: Union[Path, Dict]) -> BaseConfig:
    if isinstance(config_file_or_dict, dict):
        config_dict = config_file_or_dict.copy()
        config_dir = Path.cwd()
    else:
        with open(config_file_or_dict, "r") as f:
            config_dict = yaml.load(f, Loader=UniqueKeyLoader)
        config_dir = config_file_or_dict.parent

    trainer_type = config_dict.get("type", TRAINER_DEFAULT)
    config_dict["type"] = trainer_type

    trainer_script = config_dict.get("code", CUSTOM_CODE_DEFAULT)

    script_path = Path(trainer_script)
    if not script_path.is_absolute():
        script_path = config_dir / script_path
    script_path = script_path.resolve()

    if script_path.exists():
        config_dict["code"] = trainer_script
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
    elif config_dict.get("code") is not None:
        # User specified a script that doesn't exist
        raise FileNotFoundError(f"Cannot find script at {script_path}")

    trainer_cls = get_registered_trainer(trainer_type)
    config_cls = _get_class_attr_type_hints(trainer_cls, "config")[0]
    config = config_cls(**config_dict)

    return config
