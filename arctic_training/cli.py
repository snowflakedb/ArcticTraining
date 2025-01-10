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

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="ArticTraining config yaml file.")
    args, deepspeed_args = parser.parse_known_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")

    exe_path = shutil.which("arctic_training_run")

    env = os.environ.copy()

    subprocess.run(
        [
            "deepspeed",
            *deepspeed_args,
            exe_path,
            "--config",
            str(args.config),
        ],
        env=env,
    )


def run_script():
    from arctic_training.config.trainer import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="ArticTraining config to run.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK", 0)),
        help="Local rank of the process.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")

    config = get_config(args.config)
    trainer = config.trainer
    trainer.train()
