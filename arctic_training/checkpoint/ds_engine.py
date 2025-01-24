import random
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
import torch

from arctic_training.register import register_checkpoint

from .checkpoint import CheckpointEngine
from arctic_training.logging import logger


@register_checkpoint
class DSCheckpointEngine(CheckpointEngine):
    checkpoint_type = "deepspeed"

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoint_dir / "latest"

    @property
    def checkpoint_tag(self) -> str:
        return f"epoch_{self.trainer.epoch_idx}"

    @property
    def client_state(self) -> Dict[str, Any]:
        return {
            "epoch": self.trainer.epoch_idx,
            "train_batch_idx": self.trainer.train_batch_idx,
            "torch_random_state": torch.get_rng_state(),
            "torch_cuda_random_state": torch.cuda.get_rng_state(),
            "np_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
        }

    def save(self) -> None:
        self.model.save_checkpoint(
            self.checkpoint_dir, tag=self.checkpoint_tag, client_state=self.client_state
        )

    def load(self) -> None:
        if not self.latest_checkpoint.exists():
            logger.warning(f"Latest Checkpoint File does not exist")
            return
        _, client_states = self.model.load_checkpoint(self.checkpoint_dir)

        self.trainer.global_step_idx = self.model.global_steps
        self.trainer.epoch_idx = (
            client_states["epoch"]
        )  # TODO: make sure this is used correctly in the trainer
        self.trainer.train_batch_idx = client_states["train_batch_idx"]+1
        torch.set_rng_state(client_states["torch_random_state"])
        torch.cuda.set_rng_state(client_states["torch_cuda_random_state"])
        np.random.set_state(client_states["np_random_state"])
        random.setstate(client_states["python_random_state"])
