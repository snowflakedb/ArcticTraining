from typing import TYPE_CHECKING
from typing import List

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer

from arctic_training.callback import Callback


def callback_factory(trainer: "Trainer") -> List[Callback]:
    callbacks = [
        Callback(event=event, fn=fn) for event, fn in trainer._trainer_callbacks
    ]
    return callbacks
