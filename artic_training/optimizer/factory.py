from typing import TYPE_CHECKING
from typing import Any
from typing import Optional

from arctic_training.config import ModelConfig
from deepspeed.ops.adam import FusedAdam

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer


def optimizer_factory(
    trainer: "Trainer", model_config: Optional[ModelConfig] = None
) -> Any:
    if model_config is None:
        model_config = trainer.config.model
    optimizer_grouped_params = model_config.create_optimizer_grouped_params(
        trainer.model, trainer.config.weight_decay
    )
    optimizer = FusedAdam(
        optimizer_grouped_params,
        lr=trainer.config.learning_rate,
        betas=trainer.config.betas,
    )
    return optimizer
