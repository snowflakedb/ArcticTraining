from pathlib import Path
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import field_validator

from .base import BaseConfig
from .enums import LogLevel


class LoggerConfig(BaseConfig):
    output_dir: Optional[Path] = None
    level: LogLevel = LogLevel.INFO
    print_output_ranks: Union[Literal["*"], List[int]] = [0]
    file_output_ranks: Union[Literal["*"], List[int]] = "*"

    @field_validator("print_output_ranks", "file_output_ranks")
    @classmethod
    def validate_output_ranks(cls, value):
        from .config import _context_world_size

        world_size = _context_world_size.get()
        if value == "*":
            return list(range(world_size))
        return value
