# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os
import sys
from functools import partialmethod
from typing import Optional
from typing import Union

from deepspeed.utils import logger as ds_logger
from loguru import logger
from tqdm import tqdm
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType

    from arctic_training.config.logger import LoggerConfig

_logger_setup: bool = False


LOG_LEVEL_DEFAULT = os.getenv("AT_LOG_LEVEL", "WARNING")
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
    " Rank %s | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> -"
    " <level>{message}</level>"
    % os.getenv("RANK", "0")
)


class InterceptHandler(logging.Handler):
    """Handler to bridge Python's builtin logging messages into Loguru.

    See: https://github.com/Delgan/loguru/blob/master/README.md#entirely-compatible-with-standard-logging
    """

    @staticmethod
    def unwind_interpreter_frames_until_logging_origin() -> int:
        """Finds the relative stack frame depth of the most recent call-site to
        a logging function relative to the caller of this function.

        To do this, we unwind interpreter frames from the frame of the caller of
        this function all the way until the part of execution before we got into
        code inside the logging module itself (i.e. we go back until the filename
        associated with the frame is no longer something like
        `/usr/lib/python3.10/logging/__init__.py`).
        """
        # Jump back to the frame of the caller of this function.
        current_frame = inspect.currentframe()
        assert current_frame is not None
        caller_frame = current_frame.f_back
        assert caller_frame is not None

        # Walk back up the stack until we get out of the logging module.
        frame: Optional["FrameType"] = caller_frame
        depth = 0
        while frame:
            filename = frame.f_code.co_filename
            is_logging = filename == logging.__file__
            is_frozen = "importlib" in filename and "_bootstrap" in filename
            if depth > 0 and not (is_logging or is_frozen):
                break
            frame = frame.f_back
            depth += 1

        return depth

    def emit(self, record: logging.LogRecord) -> None:
        # Find the depth of the stack frame where the logging originated.
        log_origin_depth = self.unwind_interpreter_frames_until_logging_origin()

        # Attach origin and exception information to our logging.
        logger_opt = logger.opt(depth=log_origin_depth, exception=record.exc_info)

        # Try to recognize the level of the record in loguru terms.
        try:
            level: Union[str, int] = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Pass the log record into loguru.
        logger_opt.log(level, record.getMessage())


def redirect_builtin_logging_and_set_level(level: str) -> None:
    """Sets the level of built-in Python logging to the same as our `loguru`-based
    logging specification, then redirects all logs flowing through the root log
    handler to our `loguru` logger.
    """
    root_logger = logging.getLogger()
    root_logger.addHandler(InterceptHandler())
    root_logger.setLevel(level)


def set_dependencies_logger_level(level: str) -> None:
    ds_logger.setLevel(level)
    logging.getLogger("transformers").setLevel(level)
    logging.getLogger("torch").setLevel(level)


def setup_init_logger() -> None:
    logger.remove()
    logger.add(sys.stderr, colorize=True, format=LOG_FORMAT, level=LOG_LEVEL_DEFAULT)
    set_dependencies_logger_level(LOG_LEVEL_DEFAULT)


def setup_logger(config: "LoggerConfig") -> None:
    global _logger_setup
    if _logger_setup:
        return

    logger.remove()
    pre_init_sink = logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        format=LOG_FORMAT,
        level=config.level,
    )

    if config.file_enabled:
        log_file = config.log_file
        logger.add(log_file, colorize=False, format=LOG_FORMAT, level=config.level)
        logger.info(f"Logging to {log_file}")

    logger.remove(pre_init_sink)
    if config.print_enabled:
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            colorize=True,
            format=LOG_FORMAT,
            level=config.level,
        )
        logger.info("Logger enabled")
        redirect_builtin_logging_and_set_level(config.level)
    else:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        redirect_builtin_logging_and_set_level("ERROR")
        sys.stdout = open(os.devnull, "w")

    _logger_setup = True
    logger.info("Logger initialized")
