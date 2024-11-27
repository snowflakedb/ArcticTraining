from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from arctic_training.config import CheckpointConfig

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer


class CheckpointEngine(ABC):
    checkpoint_type: str

    def __init__(self, trainer: "Trainer", config: CheckpointConfig) -> None:
        self.trainer = trainer
        self.config = config

    @property
    def model(self) -> Any:
        return self.trainer.model

    @property
    def do_checkpoint(self) -> bool:
        if not self.config.enabled:
            return False
        return_value=False
        if (
            self.model.is_gradient_accumulation_boundary()
            and self.config.save_every_n_steps
            and self.model.global_steps > 0
        ):
            return_value = self.model.global_steps % self.config.save_every_n_steps == 0
        if self.config.save_every_n_epochs:
            return_value = return_value or (self.trainer.epoch_idx > 0) and (
                self.trainer.epoch_idx % self.config.save_every_n_epochs
            ) == 0
        if self.config.save_end_of_training:
            return_value = return_value or self.trainer.training_finished
        return return_value

    @property
    def checkpoint_dir(self) -> Path:
        checkpoint_dir = self.config.output_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    @property
    def load_checkpoint(self) -> bool:
        return self.config.auto_resume

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        pass
