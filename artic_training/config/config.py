import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import yaml
from arctic_training.config import BaseConfig
from arctic_training.register import get_config_class
from arctic_training.register import get_trainer_class
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

from .checkpoint import CheckpointConfig
from .data import DataConfig
from .enums import LRSchedType
from .model import ModelConfig
from .wandb import WandBConfig


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", -1))


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", 1))


class Config(BaseConfig):
    model: ModelConfig
    data: DataConfig
    wandb: WandBConfig = WandBConfig()
    deepspeed: Dict[str, Any] = {}
    epochs: int = Field(default=1, ge=0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    learning_rate: float = Field(default=5e-4, ge=0.0, alias="lr")
    weight_decay: float = Field(default=0.1, ge=0.0)
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_type: LRSchedType = LRSchedType.LINEAR
    gradient_accumulation_steps: int = 1
    micro_batch_size: int = Field(default=1, ge=1)
    seed: int = Field(default=42, ge=0)
    checkpoint: List[CheckpointConfig] = []
    train_iters: int = Field(default=0, ge=0)
    local_rank: int = Field(default_factory=get_local_rank, exclude=True)
    world_size: int = Field(default_factory=get_world_size, exclude=True)

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


def get_config(config_file_or_dict: Union[Path, Dict[str, Any]]) -> Config:
    if isinstance(config_file_or_dict, Path):
        with open(config_file_or_dict, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = config_file_or_dict

    if "trainer_class" not in config_dict:
        raise ValueError("`trainer_class` must be defined in input config.")
    trainer_cls = get_trainer_class(config_dict["trainer_class"])
    config_cls = get_config_class(trainer_cls.config)

    config = config_cls(**config_dict)

    return config
