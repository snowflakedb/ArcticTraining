from typing import TYPE_CHECKING
from typing import List

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer

from arctic_training.callback import Callback
from arctic_training.logging import logger


def callback_factory(trainer: "Trainer") -> List[Callback]:
    logger.info("Initializing callbacks")
    callbacks = []
    for event, fn in trainer._trainer_callbacks:
        logger.info(f"Adding callback {fn} to event {event}")
        callbacks.append(Callback(event=event, fn=fn))
    return callbacks
