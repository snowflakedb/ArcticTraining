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

from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict

from datasets import DatasetDict
from datasets import load_dataset
from datasets import load_from_disk

from arctic_training.config.data import DataSourceConfig
from arctic_training.data.source import DataSource
from arctic_training.data.utils import DatasetType


class HFDataSourceConfig(DataSourceConfig):
    name_or_path: Path
    """ Name of the dataset to load. """

    kwargs: Dict[str, Any] = {}
    """ Keyword arguments to pass to the datasets.load_dataset function. """


class HFDataSource(DataSource):
    """Base DataSource class for loading data with HuggingFace datasets library."""

    name = "huggingface"
    config: HFDataSourceConfig

    def load(self, config: HFDataSourceConfig, split: str) -> DatasetType:
        # Support loading local datasets
        if config.name_or_path.exists():
            dataset = load_from_disk(config.name_or_path.as_posix(), **config.kwargs)
            if isinstance(dataset, DatasetDict):
                dataset = dataset[split]
        else:
            dataset = load_dataset(
                str(config.name_or_path), split=split, **config.kwargs
            )

        return dataset


class UltraChat200K(HFDataSource):
    name = "HuggingFaceH4/ultrachat_200k"

    def pre_load_callback(self, split: str) -> str:
        split_map = {"train": "train_sft", "test": "test_sft"}
        return split_map.get(split, split)


class SlimOrca(HFDataSource):
    name = "Open-Orca/SlimOrca"

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        def process_example(example):
            return {
                "messages": [
                    {
                        "content": message["value"],
                        "role": {
                            "system": "system",
                            "human": "user",
                            "gpt": "assistant",
                        }[message["from"]],
                    }
                    for message in example["conversations"]
                ]
            }

        return dataset.map(
            process_example,
            num_proc=self.data_factory.config.num_proc,
            desc="Loading slim orca",
        )


class MetaMathQA(HFDataSource):
    name = "meta-math/MetaMathQA"

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        formatted_dataset = dataset.map(
            partial(
                self.instruct_format_conversation,
                query_key="query",
                response_key="response",
                source_name="MetaMathQA",
            ),
            desc="Loading meta-math",
        )
        return formatted_dataset

    @staticmethod
    def instruct_format_conversation(example, query_key, response_key, source_name):
        conversation = [
            {"role": "user", "content": example[query_key]},
            {"role": "assistant", "content": example[response_key]},
        ]
        return {
            "source": source_name,
            "messages": conversation,
        }


class MagicoderOSSInstruct75k(HFDataSource):
    name = "ise-uiuc/Magicoder-OSS-Instruct-75K"

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        formatted_dataset = dataset.map(
            partial(
                self.instruct_format_conversation,
                query_key="problem",
                response_key="solution",
                source_name="Magicoder",
            ),
            desc="Loading magicoder",
        )
        return formatted_dataset

    @staticmethod
    def instruct_format_conversation(example, query_key, response_key, source_name):
        conversation = [
            {"role": "user", "content": example[query_key]},
            {"role": "assistant", "content": example[response_key]},
        ]
        return {
            "source": source_name,
            "messages": conversation,
        }


class LMSysChat1M(HFDataSource):
    name = "lmsys/lmsys-chat-1m"

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        formatted_dataset = dataset.map(
            partial(self.vicuna_format_conversation, source_name="LMSYS-CHAT-1M"),
            desc="Loading lmsys",
        )
        return formatted_dataset

    @staticmethod
    def vicuna_format_conversation(example, source_name):
        messages = []
        for conv in example["conversation"]:
            messages.append({"role": conv["role"], "content": conv["content"]})
        return {
            "source": source_name,
            "messages": messages,
        }
