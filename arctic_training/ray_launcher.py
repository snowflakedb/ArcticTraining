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

from pathlib import Path
from typing import Any
from typing import Literal

import ray.train
import yaml
from ray.train import Checkpoint
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from snowflake.ml.runtime_cluster.cluster_manager import get_available_gpu

from arctic_training.config.trainer import get_config
from arctic_training.registry import get_registered_trainer


def arctic_train_func(train_config: dict[str, Any]) -> None:
    """
    Expected config schema:
    {
        "config_dict": dict,  # In-memory ArcticTraining config dictionary
        "mode": str,         # "train" or "process-data"
        "python_profile": str # "tottime", "cumtime", or "disable" (optional)
    }
    """
    config = get_config(train_config["config_dict"])
    trainer_cls = get_registered_trainer(name=config.type)  # type: ignore[attr-defined]

    # Define Ray Train callbacks
    def post_step_ray_report(self):
        """Report metrics to Ray Train after each step."""
        if self.gas_boundary and self.global_step % self.config.train_log_iter_interval == 0:
            metrics = {k: v for k, v in self.metrics.summary_dict.items()}
            ray.train.report(metrics=metrics)

    def post_checkpoint_ray_save(self):
        """Report checkpoint to Ray Train."""
        for engine in self.checkpoint_engines:
            if engine.do_checkpoint:
                checkpoint = Checkpoint.from_directory(str(engine.checkpoint_dir))
                ray.train.report(checkpoint=checkpoint)

    # Create a dynamic subclass with Ray callbacks injected
    # Dynamically name the class to reflect the base trainer (e.g., CausalTrainer -> RayCausalTrainer)
    ray_trainer_cls = type(
        f"Ray{trainer_cls.__name__}",
        (trainer_cls,),
        {
            "name": trainer_cls.name + "_ray",
            "callbacks": [
                ("post-step", post_step_ray_report),
                ("post-checkpoint", post_checkpoint_ray_save),
            ],
        },
    )

    trainer = ray_trainer_cls(config, mode=train_config["mode"])
    if train_config["mode"] == "train":

        def train():
            trainer.train()

        local_rank = ray.train.get_context().get_local_rank()
        if train_config.get("python_profile", "disable") == "disable" or local_rank != 0:
            train()
        else:
            # run profiler on rank 0
            # XXX: how do we prevent it from running on other nodes?
            import cProfile
            from pstats import SortKey

            sort_key = (
                SortKey.TIME if train_config.get("python_profile", "disable") == "tottime" else SortKey.CUMULATIVE
            )
            cProfile.runctx("train()", {}, locals(), sort=sort_key)


def launch(
    config_file: str,
    mode: Literal["train", "process-data"],
    python_profile: Literal["tottime", "cumtime", "disable"] = "disable",
):
    # Load config from file and pass the in-memory dict to workers
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    train_config = {
        "config_dict": config_dict,
        "mode": mode,
        "python_profile": python_profile,
    }

    # TODO: Add fallback handling for when GPUs aren't available
    num_gpus = int(get_available_gpu())
    trainer = TorchTrainer(
        arctic_train_func,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(num_workers=num_gpus, use_gpu=True),
    )

    result = trainer.fit()
    return result


if __name__ == "__main__":
    import argparse
    import textwrap

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
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")

    result = launch(
        config_file=str(args.config),
        mode=args.mode,
        python_profile=args.python_profile,
    )
    print(result)
