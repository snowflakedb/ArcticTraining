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

from datasets import Dataset
from datasets import load_dataset

from arctic_training.data.source import DataSource
from arctic_training.registry import register


@register
class UltraChat200K(DataSource):
    name = "HuggingFaceH4/ultrachat_200k"
    data_factory_type = "sft"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        return load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="test_sft" if eval else "train_sft",
            num_proc=num_proc,
        ).select_columns(["messages"])


@register
class SlimOrca(DataSource):
    name = "Open-Orca/SlimOrca"
    data_factory_type = "sft"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        if eval:
            assert False, "Test split does not exist."
        dataset = load_dataset("Open-Orca/SlimOrca", split="train", num_proc=num_proc)

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

        return dataset.map(process_example, num_proc=num_proc, desc="Loading slim orca")


@register
class MetaMathQA(DataSource):
    name = "meta-math/MetaMathQA"
    data_factory_type = "sft"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        if eval:
            assert False, "Test split does not exist."
        dataset = load_dataset("meta-math/MetaMathQA", split="train", num_proc=num_proc)
        formatted_dataset = dataset.map(
            partial(
                self.instruct_format_conversation,
                query_key="query",
                response_key="response",
                source_name="MetaMathQA",
            ),
            desc="Loading meta-math",
        )
        return formatted_dataset.select_columns(["messages"])

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


@register
class MagicoderOSSInstruct75k(DataSource):
    name = "ise-uiuc/Magicoder-OSS-Instruct-75K"
    data_factory_type = "sft"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        if eval:
            assert False, "Test split does not exist."
        dataset = load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            num_proc=num_proc,
        )

        formatted_dataset = dataset.map(
            partial(
                self.instruct_format_conversation,
                query_key="problem",
                response_key="solution",
                source_name="Magicoder",
            ),
            desc="Loading magicoder",
        )
        return formatted_dataset.select_columns(["messages"])

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


@register
class LMSysChat1M(DataSource):
    name = "lmsys/lmsys-chat-1m"
    data_factory_type = "sft"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        if eval:
            assert False, "Test split does not exist."
        dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", num_proc=num_proc)
        formatted_dataset = dataset.map(
            partial(self.vicuna_format_conversation, source_name="LMSYS-CHAT-1M"),
            desc="Loading lmsys",
        )
        return formatted_dataset.select_columns(["messages"])

    @staticmethod
    def vicuna_format_conversation(example, source_name):
        messages = []
        for conv in example["conversation"]:
            messages.append({"role": conv["role"], "content": conv["content"]})
        return {
            "source": source_name,
            "messages": messages,
        }
