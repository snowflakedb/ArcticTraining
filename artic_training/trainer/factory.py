from arctic_training.config.config import Config
from arctic_training.register import get_trainer_class
from arctic_training.trainer.trainer import Trainer


def trainer_factory(config: Config) -> Trainer:
    trainer_cls = get_trainer_class(config)
    return trainer_cls(config)
