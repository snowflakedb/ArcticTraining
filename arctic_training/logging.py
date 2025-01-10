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

import logging
import os
import sys
from functools import partialmethod

from deepspeed.utils import logger as ds_logger
from loguru import logger
from tqdm import tqdm
from typing_extensions import TYPE_CHECKING

from .utils import get_local_rank

if TYPE_CHECKING:
    from arctic_training.config.logger import LoggerConfig

_logger_setup: bool = False


LOG_LEVEL_DEFAULT = os.getenv("AT_LOG_LEVEL", "WARNING")
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> |"
    " Rank %d | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> -"
    " <level>{message}</level>"
    % get_local_rank()
)


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
        set_dependencies_logger_level(config.level)
    else:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
        set_dependencies_logger_level("ERROR")
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    _logger_setup = True
    logger.info("Logger initialized")
