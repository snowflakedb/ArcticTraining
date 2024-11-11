from .base import BaseConfig


class WandBConfig(BaseConfig):
    enable: bool = False
    project: str = "snowflake"
