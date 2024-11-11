from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from arctic_training.trainer import Trainer


class Callback:
    def __init__(self, event: str, fn: Callable) -> None:
        self.event = event
        self.fn = fn

    def __call__(self, trainer: "Trainer") -> None:
        self.fn(trainer)
