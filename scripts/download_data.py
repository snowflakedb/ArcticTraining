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

import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


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
    sources = config["data"]["sources"]
    for source in sources:
        if isinstance(source, dict):
            name_or_path = source["name_or_path"]
            split = source.get("split", None)
            kwargs = source.get("kwargs", {})
        else:
            name_or_path = source
            split = None
            kwargs = {}
        print(f"{name_or_path=}, {split=}, {kwargs=}")
        load_dataset(path=str(name_or_path), split=split, **kwargs)


def download_model(config):
    name_or_path = config["model"]["name_or_path"]
    print(f"Downloading model: {name_or_path=}")
    AutoModelForCausalLM.from_pretrained(name_or_path)
    AutoTokenizer.from_pretrained(name_or_path)


def get_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    args = get_args()

    config = get_config(args.config)

    if args.download == "data":
        tempfile.tempdir = args.tmp_path
        os.makedirs(args.tmp_path, exist_ok=True)
        download_data(config)
    elif args.download == "model":
        download_model(config)
    else:
        raise ValueError(f"Invalid download type: {args.download}")
