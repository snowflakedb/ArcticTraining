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

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from arctic_training.config.wandb import WandBConfig


def test_wandb_callback():
    pytest.skip("Skipping test for now until we refactor to use a proper DummyTrainer class")
    os.environ["WANDB_MODE"] = "offline"
    wandb_config = WandBConfig(
        enable=True,
        project="test_project",
    )

    # TODO: Make a DummyTrainer class that can be used in multiple tests
    class DummyTrainer:
        config = SimpleNamespace(model_dump=lambda: {}, wandb=wandb_config)
        model = SimpleNamespace(lr_scheduler=SimpleNamespace(get_last_lr=lambda: [0.1]))
        global_step = 0
        global_rank = wandb_config.global_rank

    # trainer = DummyTrainer()

    # init_wandb_project(trainer)
    # log_wandb_loss(trainer, 0.1)
    # teardown_wandb(trainer)

    output_path = list(Path("./wandb/").glob("offline-run-*/run-*.wandb"))[0]
    assert output_path, "No wandb file found"
