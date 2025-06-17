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
import shutil
import textwrap
from pathlib import Path

from deepspeed.launcher.runner import main as ds_runner


def main():
    parser = argparse.ArgumentParser(
        prog="arctic_training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            DeepSpeed Args:
                ArcticTraining uses the DeepSpeed launcher to create a
                distributed training environment. Any additional args after the
                config file path will be passed directly to the DeepSpeed
                launcher.

                For example, `arctic_training my_config.yaml --num_gpus 2`.

                To see a full list of DeepSpeed launcher args, run `deepspeed --help`.
            """
        ),
    )
    parser.add_argument(
        "mode",
        type=str,
        nargs="?",
        choices=["train", "process-data"],
        default="train",
        help="Operation mode, 'process-data' will run the data processing pipeline.",
    )
    parser.add_argument("config", type=Path, help="ArticTraining config yaml file.")
    args, deepspeed_args = parser.parse_known_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")

    exe_path = shutil.which("arctic_training_run")

    ds_runner(
        [
            *deepspeed_args,
            exe_path,
            "--mode",
            args.mode,
            "--config",
            str(args.config),
        ]
    )


if __name__ == "__main__":
    main()
