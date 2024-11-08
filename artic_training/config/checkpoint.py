from pathlib import Path

from arctic_training.config import BaseConfig
from arctic_training.register import get_checkpoint_class
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self


class CheckpointConfig(BaseConfig):
    type: str = "deepspeed"
    output_dir: Path
    enabled: bool = True
    auto_resume: bool = False
    save_every_n_steps: int = Field(default=0, ge=0)
    save_every_n_epochs: int = Field(default=0, ge=0)
    save_end_of_training: bool = False

    @model_validator(mode="after")
    def validate_checkpoint_type(self) -> Self:
        _ = get_checkpoint_class(self.type)
        return self
