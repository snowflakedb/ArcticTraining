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
import textwrap
from pathlib import Path


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
        check=True,
    )

def wait_forever():
    import time
    import sys
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


def run_model():
    from arctic_training.utils import send_dict, recv_dict
    from transformers import AutoModelForCausalLM
    import sys
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    try:
        while True:
            for i in range(8):
                inputs = recv_dict(src=i)
                outputs = model.generate(**inputs, max_length=50)
                send_dict(outputs, dst=i)
    except KeyboardInterrupt:
        pass


def run_script():
    import deepspeed.comm as dist

    from arctic_training.config.trainer import get_config
    from arctic_training.registry import get_registered_trainer

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

    if config.global_rank > 8:
        wait_forever()
    elif config.global_rank == 8:
        run_model()
    else:
        trainer_cls = get_registered_trainer(name=config.type)
        trainer = trainer_cls(config)
        trainer.train()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
