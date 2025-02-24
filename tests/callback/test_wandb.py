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
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

from arctic_training.callback.wandb import init_wandb_project
from arctic_training.callback.wandb import log_wandb_loss
from arctic_training.callback.wandb import teardown_wandb
from arctic_training.config.wandb import WandBConfig


def test_wandb_callback():
    wandb_config = WandBConfig(
        enable=True,
        project="test_project",
    )

    os.environ["WANDB_MODE"] = "offline"

    # TODO: Make a DummyTrainer class that can be used in multiple tests
    class DummyTrainer:
        config = SimpleNamespace(model_dump=lambda: {}, wandb=wandb_config)
        model = SimpleNamespace(lr_scheduler=SimpleNamespace(get_last_lr=lambda: [0.1]))
        global_step = 0
        global_rank = wandb_config.global_rank

    trainer = DummyTrainer()

    expected_loss = 0.1
    init_wandb_project(trainer)
    log_wandb_loss(trainer, expected_loss)
    teardown_wandb(trainer)

    output_path = list(Path("./wandb/").glob("offline-run-*/run-*.wandb"))[0]
    assert output_path, "No wandb file found"

    content = subprocess.check_output(
        f"wandb sync --view --verbose {output_path} | grep 'train/loss' -A 1 | tail"
        " -n 1",
        shell=True,
    )
    recorded_loss = float(
        re.findall(r"value_json: \"(\d+\.\d+)\"", content.decode())[0]
    )
    assert (
        recorded_loss == expected_loss
    ), f"Expected loss: {expected_loss}, got: {recorded_loss}"
