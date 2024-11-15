from loguru import logger
from tqdm import tqdm

from arctic_training.config import Config


def setup_logger(config: Config):
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | Rank %d | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        % config.local_rank
    )
    logger.remove()
    pre_init_sink = logger.add(
        lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format
    )

    if (
        config.local_rank in config.logger.file_output_ranks
        and config.logger.output_dir
    ):
        log_file = config.logger.output_dir / f"log_{config.local_rank}.log"
        logger.add(log_file, colorize=False, format=log_format)
        logger.info(f"Logging to {log_file}")

    logger.remove(pre_init_sink)
    if config.local_rank in config.logger.print_output_ranks:
        logger.add(
            lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format
        )
        logger.info("Logger enabled")

    logger.info("Logger initialized")
