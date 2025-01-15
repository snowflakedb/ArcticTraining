import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class TorchInstallHook(BuildHookInterface):
    def initialize(self, version, build_data):
        try:
            import torch  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            subprocess.run(["pip", "install", "torch"], check=True)
