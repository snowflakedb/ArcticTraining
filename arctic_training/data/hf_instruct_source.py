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

from typing import Dict

from pydantic import Field

from arctic_training.data.hf_source import HFDataSource
from arctic_training.data.hf_source import HFDataSourceConfig
from arctic_training.data.utils import DatasetType


class HFInstructDataSourceConfig(HFDataSourceConfig):
    role_mapping: Dict[str, str] = Field(default_factory=lambda: {"user": "user", "assistant": "assistant"})
    """
    Map dataset columns to message roles OR map role field values to standard roles.
    For column mapping: {"question": "user", "response": "assistant"}
    For role value mapping: {"human": "user", "ai": "assistant"}
    """

    conversation_column: str = "messages"
    """Column name containing pre-formatted conversation arrays."""

    role_field: str = "role"
    """Field name for role in conversation dicts."""

    content_field: str = "content"
    """Field name for content in conversation dicts."""


class HFInstructDataSource(HFDataSource):
    """Base DataSource class for instruction-following datasets used in SFT."""

    name = "huggingface_instruct"
    config: HFInstructDataSourceConfig

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        if self.config.conversation_column in dataset.column_names:
            dataset = self._format_conversation_column(dataset)
        elif all(col in dataset.column_names for col in self.config.role_mapping.keys()):
            dataset = self._format_role_mapping(dataset)
        else:
            raise ValueError(
                "Dataset does not contain expected columns. "
                f"Available columns: {sorted(dataset.column_names)}. "
                f"Expected either conversation column '{self.config.conversation_column}' "
                f"or role mapping columns: {sorted(self.config.role_mapping.keys())}."
            )

        return dataset

    def _format_conversation_column(self, dataset: DatasetType) -> DatasetType:
        """Format dataset when using a pre-existing conversation column."""

        def process_example(example):
            conversation = example[self.config.conversation_column]
            messages = []

            for item in conversation:
                role = item.get(self.config.role_field)
                content = item.get(self.config.content_field)
                if role in self.config.role_mapping:
                    role = self.config.role_mapping[role]
                messages.append({"role": role, "content": content})

            return {"messages": messages}

        return dataset.map(
            process_example,
            num_proc=self.data_factory.config.num_proc,
            desc=f"Loading {self.name}",
        )

    def _format_role_mapping(self, dataset: DatasetType) -> DatasetType:
        """Format dataset when using role mapping."""

        def process_example(example):
            messages = []

            for column, role in self.config.role_mapping.items():
                messages.append({"role": role, "content": example[column]})

            return {"messages": messages}

        return dataset.map(
            process_example,
            num_proc=self.data_factory.config.num_proc,
            desc=f"Loading {self.name}",
        )
