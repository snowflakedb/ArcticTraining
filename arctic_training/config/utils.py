import os


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", -1))


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", 1))
