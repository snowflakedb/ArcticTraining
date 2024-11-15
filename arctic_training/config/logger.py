from pathlib import Path
from typing import List
from typing import Literal
from typing import Union

from pydantic import computed_field
from pydantic import field_validator

from .base import BaseConfig
from .enums import LogLevel
from .utils import get_local_rank
from .utils import get_world_size


class LoggerConfig(BaseConfig):
    output_dir: Path = Path("/dev/null")
    level: LogLevel = LogLevel.INFO
    print_output_ranks: Union[Literal["*"], List[int]] = [0]
    file_output_ranks: Union[Literal["*"], List[int]] = "*"

    @field_validator("print_output_ranks", "file_output_ranks", mode="before")
    def fill_output_ranks(cls, v):
        if v == "*":
            return range(get_world_size())
        return v

    @property
    @computed_field
    def log_file(self) -> Path:
        local_rank = get_local_rank()
        return self.output_dir / f"rank_{local_rank}.log"

    @property
    @computed_field
    def file_enabled(self) -> bool:
        local_rank = get_local_rank()
        return local_rank in self.file_output_ranks

    @property
    @computed_field
    def print_enabled(self) -> bool:
        local_rank = get_local_rank()
        return local_rank in self.print_output_ranks
