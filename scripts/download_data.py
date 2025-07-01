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
import tempfile
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from arctic_training.config.trainer import get_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="ArticTraining config to run.",
    )

    parser.add_argument(
        "-d",
        "--download",
        type=str,
        choices=["data", "model"],
        required=True,
        help="Download data sources or model weights.",
    )

    parser.add_argument(
        "-t",
        "--tmp-path",
        type=Path,
        default=Path("/data-fast/data-tmp"),
        help="Path to a temporary directory to download data to.",
    )
    return parser.parse_args()


def download_data(config):
    for source in config.data.sources:
        print(f"{source.name_or_path=}, {source.split=}, {source.kwargs=}")
        load_dataset(path=str(source.name_or_path), split=source.split, **source.kwargs)


def download_model(config):
    print(f"Downloading model: {config.model.name_or_path=}")
    AutoModelForCausalLM.from_pretrained(config.model.name_or_path)
    AutoTokenizer.from_pretrained(config.model.name_or_path)


if __name__ == "__main__":
    args = get_args()

    # FIXME(jeff): currently `get_config` requires a distributed environment, we should
    # be able to parse a trainer config yaml without having to initialize torch distributed.
    # This is a hack to get around this for now.
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    tempfile.tempdir = args.tmp_path
    os.makedirs(args.tmp_path, exist_ok=True)

    config = get_config(args.config)

    if args.download == "data":
        download_data(config)
    elif args.download == "model":
        download_model(config)
    else:
        raise ValueError(f"Invalid download type: {args.download}")
