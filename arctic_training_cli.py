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
import textwrap
from pathlib import Path


def deepspeed_launch(config_file: str, mode: str, python_profile: str, deepspeed_args: list[str]):
    import deepspeed
    from deepspeed.launcher.runner import main as ds_runner

    deepspeed.launcher.runner.EXPORT_ENVS = deepspeed.launcher.runner.EXPORT_ENVS + [
        "WANDB",
    ]

    # If CUDA_VISIBLE_DEVICES is set and --num_gpus is not specified,
    # automatically set --num_gpus to match the visible devices.
    # NOTE: CUDA_VISIBLE_DEVICES must start from GPU 0 (e.g., 0,1,2) because
    # DeepSpeed's launcher always remaps to 0-indexed GPUs. If you need to
    # reserve GPU 0 for another process (like a vLLM teacher), put that process
    # on the LAST GPU instead (e.g., teacher on GPU 3, training on GPUs 0,1,2).
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    has_num_gpus = any(arg.startswith("--num_gpus") or arg == "-n" for arg in deepspeed_args)
    has_include = any(arg.startswith("--include") for arg in deepspeed_args)
    has_exclude = any(arg.startswith("--exclude") for arg in deepspeed_args)

    if cuda_visible and not has_num_gpus and not has_include and not has_exclude:
        num_visible_gpus = len([g for g in cuda_visible.split(",") if g.strip()])
        deepspeed_args = ["--num_gpus", str(num_visible_gpus)] + list(deepspeed_args)

    runner_name = "arctic_training_run"
    exe_path = shutil.which(runner_name)
    if exe_path is None:
        raise ValueError(f"can't find {runner_name} in paths of env var PATH={os.environ['PATH']}")

    return ds_runner(
        [
            *deepspeed_args,
            exe_path,
            "--mode",
            mode,
            "--config",
            config_file,
            "--python_profile",
            python_profile,
        ]
    )


def ray_launch(config_file: str, mode: str, python_profile: str):
    from arctic_training.launcher.ray_launcher import launch as ray_launch

    return ray_launch(
        config_file=config_file,
        mode=mode,
        python_profile=python_profile,
    )


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
    parser.add_argument(
        "--python_profile",
        type=str,
        choices=["tottime", "cumtime", "disable"],
        default="disable",
        help=(
            "Train under Python profile. Sort results by tottime or cumtime. This is an experimental feature and the"
            " API is likely to change"
        ),
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["deepspeed", "ray"],
        default=None,
        help="The launcher to use for distributed training. Defaults to deepspeed.",
    )
    args, deepspeed_args = parser.parse_known_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")

    # CLI --launcher flag takes precedence over USE_RAY environment variable
    use_ray_env = os.environ.get("USE_RAY", "").lower() in ("1", "true")
    use_ray = args.launcher == "ray" or (args.launcher is None and use_ray_env)
    if use_ray:
        if len(deepspeed_args) > 0:
            raise ValueError("DeepSpeed arguments are not supported when using Ray launcher.")
        ray_launch(config_file=str(args.config), mode=args.mode, python_profile=args.python_profile)
        return

    deepspeed_launch(
        config_file=str(args.config), mode=args.mode, python_profile=args.python_profile, deepspeed_args=deepspeed_args
    )


if __name__ == "__main__":
    main()
