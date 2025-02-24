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

from typing import TYPE_CHECKING

import torch

import wandb

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


def init_wandb_project(self: "Trainer") -> None:
    if self.global_rank == 0 and self.config.wandb.enable:
        # Note: wandb.init() is not type annotated so we need to use type: ignore
        self.wandb_experiment = wandb.init(  # type: ignore
            project=self.config.wandb.project, config=self.config.model_dump()
        )


def log_wandb_loss(self: "Trainer", loss: torch.Tensor) -> torch.Tensor:
    if self.wandb_experiment is not None:
        self.wandb_experiment.log(
            {
                "train/loss": loss,
                "train/lr": self.model.lr_scheduler.get_last_lr()[0],
                "global_step": self.global_step,
            }
        )
    return loss


init_wandb_project_cb = ("post-init", init_wandb_project)
log_wandb_loss_cb = ("post-loss", log_wandb_loss)
