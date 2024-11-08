from typing import TYPE_CHECKING
from typing import List
from typing import Optional

from arctic_training.checkpoint import CheckpointEngine
from arctic_training.config import CheckpointConfig
from arctic_training.register import get_checkpoint_class

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer


def checkpoint_factory(
    trainer: "Trainer", checkpoint_configs: Optional[List[CheckpointConfig]] = None
) -> List[CheckpointEngine]:
    if checkpoint_configs is None:
        checkpoint_configs = trainer.config.checkpoint
    if isinstance(checkpoint_configs, CheckpointConfig):
        checkpoint_configs = [checkpoint_configs]

    checkpoint_engines = []
    for checkpoint_config in checkpoint_configs:
        checkpoint_engine = get_checkpoint_class(checkpoint_config.type)(
            trainer, checkpoint_config
        )
        checkpoint_engines.append(checkpoint_engine)

    return checkpoint_engines
