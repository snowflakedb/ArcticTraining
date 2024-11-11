from typing import TYPE_CHECKING
from typing import Any

from transformers import get_scheduler

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer


def scheduler_factory(trainer: "Trainer") -> Any:
    lr_scheduler = get_scheduler(
        name=trainer.config.lr_scheduler_type,
        optimizer=trainer.optimizer,
        num_warmup_steps=trainer.warmup_steps,
        num_training_steps=trainer.training_horizon,
    )
    return lr_scheduler
