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
from typing import Any
from typing import Dict

from datasets import load_dataset
from pydantic import model_validator
from typing_extensions import Self

from arctic_training.config.data import DataSourceConfig
from arctic_training.data.source import DataSource
from arctic_training.data.utils import DatasetType
from arctic_training.logging import logger
from arctic_training.registry import get_registered_data_source


class HFDataSourceConfig(DataSourceConfig):
    dataset_name: str = ""
    """ Name of the dataset to load. """

    kwargs: Dict[str, Any] = {}
    """ Keyword arguments to pass to the datasets.load_dataset function. """

    @model_validator(mode="after")
    def set_dataset_name(self) -> Self:
        if self.dataset_name == "":
            try:
                data_source = get_registered_data_source(name=self.type)
                logger.warning(
                    f"No dataset name was provided for {data_source.name}. Auto-filling"
                    " value based on selected dataset for backwargs compatibility."
                    " However this feature will be removed in a future version of"
                    " ArcticTraining."
                )
                if data_source.name == "huggingface":
                    raise ValueError(
                        "Must provide a dataset name for HuggingFace data sources."
                    )
                self.dataset_name = data_source.name
            except ValueError as e:
                logger.error(
                    "No dataset name was provided and failed to infer one from data"
                    f" source type {self.type}."
                )
                raise e
        return self


class HFDataSource(DataSource):
    """Base DataSource class for loading data with HuggingFace datasets library."""

    name = "huggingface"
    config: HFDataSourceConfig

    def load(self, config: HFDataSourceConfig, split: str) -> DatasetType:
        return load_dataset(config.dataset_name, split=split, **config.kwargs)


class UltraChat200K(HFDataSource):
    name = "HuggingFaceH4/ultrachat_200k"

    def pre_load_callback(self, split: str) -> str:
        split_map = {"train": "train_sft", "test": "test_sft"}
        return split_map.get(split, split)


# XXX: need an easier way to create truncated datasets of desired length
from datasets import Dataset
from datasets import IterableDataset
def modify_config_for_truncated_data(self):
    self.config.kwargs["streaming"] = True  # Avoid downloading entire dataset
    self.config.dataset_name = self.name.removesuffix(  # Set to the real dataset name
        "-10k"
    )


def sample_data_for_truncated_dataset(self, dataset: IterableDataset) -> Dataset:
    return Dataset.from_list(list(dataset.take(10000)), features=dataset.features)


class UltraChat10k(UltraChat200K):
    name = "HuggingFaceH4/ultrachat_200k-10k"
    callbacks = [
        ("post-init", modify_config_for_truncated_data),
        ("post-load", sample_data_for_truncated_dataset),
    ]

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
