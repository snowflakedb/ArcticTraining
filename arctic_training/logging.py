from loguru import logger
from tqdm import tqdm
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from arctic_training.config import Config


def setup_logger(config: "Config"):
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | Rank %d | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        % config.local_rank
    )
    logger.remove()
    pre_init_sink = logger.add(
        lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format
    )

    if config.logger.file_enabled:
        log_file = config.logger.log_file
        logger.add(log_file, colorize=False, format=log_format)
        logger.info(f"Logging to {log_file}")

    logger.remove(pre_init_sink)
    if config.logger.print_enabled:
        logger.add(
            lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format
        )
        logger.info("Logger enabled")

    logger.info("Logger initialized")
