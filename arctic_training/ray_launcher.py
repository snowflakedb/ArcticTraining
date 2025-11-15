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
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from snowflake.ml.runtime_cluster.cluster_manager import get_available_gpu

from arctic_training.config.trainer import get_config
from arctic_training.registry import get_registered_trainer


def arctic_train_func(train_config: dict[str, Any]) -> None:
    """
    Expected config schema:
    {
        "config_file": Path,  # Path to ArticTraining config file
        "mode": str,         # "train" or "process-data"
        "python_profile": str # "tottime", "cumtime", or "disable" (optional)
    }
    """
    if not train_config["config_file"].exists():
        raise FileNotFoundError(f"Config file {train_config['config_file']} not found.")

    config = get_config(train_config["config_file"])
    trainer_cls = get_registered_trainer(name=config.type)
    trainer = trainer_cls(config, mode=train_config["mode"])
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
            cProfile.runctx("train()", None, locals(), sort=sort_key)


def launch(
    config_file: str,
    mode: Literal["train", "process-data"],
    python_profile: Literal["tottime", "cumtime", "disable"] = "disable",
):
    train_config = {
        "config_file": Path(config_file).absolute(),
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

    # TODO: Do something more interesting with the results
    print(result)


if __name__ == "__main__":
    launch("run-causal.yml", mode="train")
