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
from typing import Callable
from typing import Literal
from typing import cast

import ray
import ray.train
import yaml
from ray.train import Checkpoint
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

from arctic_training.callback.callback import Callback
from arctic_training.config.trainer import TrainerConfig
from arctic_training.config.trainer import get_config
from arctic_training.registry import get_registered_trainer
from arctic_training.trainer.trainer import Trainer

TrainConfig = dict[str, Any]


def _post_step_ray_report(trainer: Trainer) -> None:
    """Report metrics to Ray Train after each training step."""
    if trainer.gas_boundary and trainer.global_step % trainer.config.train_log_iter_interval == 0:
        metrics = {k: v for k, v in trainer.metrics.summary_dict.items()}
        ray.train.report(metrics=metrics)


def _post_checkpoint_ray_save(trainer: Trainer) -> None:
    """Upload checkpoints produced by the trainer to Ray Train."""
    for engine in trainer.checkpoint_engines:
        if engine.do_checkpoint:
            checkpoint = Checkpoint.from_directory(str(engine.checkpoint_dir))
            ray.train.report(checkpoint=checkpoint)


def _attach_ray_callbacks(trainer: Trainer) -> None:
    """Append Ray reporting callbacks to an instantiated trainer without creating a new subclass."""
    event_methods = trainer._get_all_callback_event_methods()
    callbacks_to_add = []

    if "post-step" in event_methods:
        callbacks_to_add.append(Callback("post-step", _post_step_ray_report, event_methods["post-step"]))
    if "post-checkpoint" in event_methods:
        callbacks_to_add.append(
            Callback("post-checkpoint", _post_checkpoint_ray_save, event_methods["post-checkpoint"])
        )

    trainer._initialized_callbacks.extend(callbacks_to_add)


def get_available_gpu() -> int:
    """
    Get the number of available GPUs for current ray cluster.

    Returns:
        int: number of available GPUs.
    """
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)
    return int(ray.available_resources().get("GPU", 0))


def make_arctic_train_func() -> Callable[[TrainConfig], None]:
    """
    Build a Ray Train-compatible training loop function.

    The returned callable is a closure that Ray will cloudpickle and execute
    on each worker node. It expects a ``train_config`` dictionary with the
    following schema:

    {
        "arctic_config": dict,   # In-memory ArcticTraining config dictionary
        "mode": str,           # "train" or "process-data"
        "python_profile": str, # "tottime", "cumtime", or "disable" (optional)
    }
    """

    def _maybe_profile(train_fn: Callable[[], None], python_profile: str) -> None:
        """
        Optionally run ``train_fn`` under a Python profiler for local rank 0.
        TODO: Dedupe with entrypoint.py
        """
        local_rank = ray.train.get_context().get_local_rank()
        if python_profile == "disable" or local_rank != 0:
            train_fn()
            return

        # Run profiler on rank 0 only
        # XXX: how do we prevent it from running on other nodes?
        import cProfile
        from pstats import SortKey

        sort_key = SortKey.TIME if python_profile == "tottime" else SortKey.CUMULATIVE
        cProfile.runctx("train_fn()", {}, locals(), sort=sort_key)

    def arctic_train_func(train_config: TrainConfig) -> None:
        config = cast(TrainerConfig, get_config(train_config["arctic_config"]))
        trainer_cls = get_registered_trainer(name=config.type)
        trainer = trainer_cls(config, mode=train_config["mode"])
        _attach_ray_callbacks(trainer)

        if train_config["mode"] != "train":
            # For "process-data" mode (or any non-train mode) we simply construct the trainer.
            return

        python_profile = train_config.get("python_profile", "disable")
        _maybe_profile(trainer.train, python_profile)

    return arctic_train_func


def launch(
    config_file: str,
    mode: Literal["train", "process-data"],
    python_profile: Literal["tottime", "cumtime", "disable"] = "disable",
):
    """
    Entry point for launching training via Ray.

    - Loads the training config from ``config_file``
    - Constructs a Ray ``TorchTrainer`` with GPU-based scaling
    - Uses a closure-wrapped ``arctic_train_func`` that Ray can cloudpickle
    """
    # Load config from file and pass the in-memory dict to workers
    with open(config_file, "r") as f:
        arctic_config = yaml.safe_load(f)

    train_config: dict[str, Any] = {
        "arctic_config": arctic_config,
        "mode": mode,
        "python_profile": python_profile,
    }

    num_gpus = get_available_gpu()
    use_gpu = num_gpus > 0
    num_workers = num_gpus if use_gpu else 1
    arctic_train_func = make_arctic_train_func()

    trainer = TorchTrainer(
        arctic_train_func,
        train_loop_config=train_config,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    )

    return trainer.fit()


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
