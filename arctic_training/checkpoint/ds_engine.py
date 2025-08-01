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

import random
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
import torch

from arctic_training.checkpoint.engine import CheckpointEngine


def model_norm(model: torch.nn.Module) -> float:
    # Caution: might cause OOM if using with large models, ideally only used in debugging small models
    with torch.no_grad():
        all_params = torch.cat([p.view(-1) for p in model.parameters()])
        return torch.norm(all_params, p=2).item()


def optim_state_norm(optimizer: torch.optim.Optimizer) -> float:
    # Caution: might cause OOM if using with large models, ideally only used in debugging small models
    with torch.no_grad():
        all_state_tensors = []
        for state in optimizer.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    all_state_tensors.append(v.view(-1))
        if not all_state_tensors:
            return 0.0  # No state yet (e.g. before first step)
        flat = torch.cat(all_state_tensors)
        return torch.norm(flat, p=2).item()


class DSCheckpointEngine(CheckpointEngine):
    name = "deepspeed"

    @property
    def checkpoint_dir(self) -> Path:
        """Returns the directory where the checkpoint will be saved.

        Override the base class implementation to avoid the extra global_step_X
        subdirectory since DeepSpeed handles versioning through its own tagging system.
        """
        checkpoint_dir = self.config.output_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    @property
    def latest_checkpoint(self) -> Path:
        return self.checkpoint_dir / "latest"

    @property
    def checkpoint_tag(self) -> str:
        return f"epoch_{self.trainer.epoch_idx}_global_step_{self.trainer.global_step}"

    @property
    def client_state(self) -> Dict[str, Any]:
        state = {
            "end_of_epoch": self.trainer.epoch_idx,
            "torch_random_state": torch.get_rng_state(),
            "np_random_state": np.random.get_state(),
            "python_random_state": random.getstate(),
            "global_step": self.trainer.global_step,
        }
        if self.device != torch.device("cpu"):
            state["torch_cuda_random_state"] = torch.cuda.get_rng_state()
        return state

    def save(self, model) -> None:
        model.save_checkpoint(
            self.checkpoint_dir,
            tag=self.checkpoint_tag,
            client_state=self.client_state,
        )

        # Helpful ckpt resume debugging snippet
        # norm = model_norm(model)
        # optim_norm = optim_state_norm(model.optimizer.optimizer)
        # logger.info(f"Saved model norm: {norm}, optim norm: {optim_norm}")

    def load(self, model) -> None:
        if not self.latest_checkpoint.exists():
            return
        _, client_states = model.load_checkpoint(self.checkpoint_dir)

        self.trainer.global_step = client_states["global_step"]
        # align deepspeed global_steps with arctic_training global_step
        model.global_steps = self.trainer.global_step
        self.trainer.epoch_idx = client_states["end_of_epoch"]
        torch.set_rng_state(client_states["torch_random_state"])
        np.random.set_state(client_states["np_random_state"])
        random.setstate(client_states["python_random_state"])
        if self.device != torch.device("cpu"):
            torch.cuda.set_rng_state(client_states["torch_cuda_random_state"])

        # Helpful ckpt resume debugging snippet
        # norm = model_norm(model)
        # optim_norm = optim_state_norm(model.optimizer.optimizer)
        # logger.info(f"Loaded model norm: {norm}, optim norm: {optim_norm}")
