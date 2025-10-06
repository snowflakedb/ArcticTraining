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
from pathlib import Path

import deepspeed.comm as dist

from arctic_training.config.trainer import get_config
from arctic_training.registry import get_registered_trainer


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "process-data"],
        default="train",
        help="Operation mode, 'process-data' will run the data processing pipeline.",
    )
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

    config = get_config(args.config)
    trainer_cls = get_registered_trainer(name=config.type)
    trainer = trainer_cls(config, mode=args.mode)
    if args.mode == "train":

        def train():
            trainer.train()

        if args.python_profile == "disable" or args.local_rank != 0:
            train()
        else:
            # run profiler on rank 0
            # XXX: how do we prevent it from running on other nodes?
            import cProfile
            from pstats import SortKey

            sort_key = SortKey.TIME if args.python_profile == "tottime" else SortKey.CUMULATIVE
            cProfile.runctx("train()", None, locals(), sort=sort_key)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
