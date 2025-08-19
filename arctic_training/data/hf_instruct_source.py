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

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

from arctic_training.data.hf_source import HFDataSource
from arctic_training.data.hf_source import HFDataSourceConfig
from arctic_training.data.utils import DatasetType

# Known datasets with their default role mappings
KNOWN_HF_INSTRUCT_DATASETS: Dict[str, Dict[str, Any]] = {
    "nvidia/AceMath-Instruct-Training-Data": dict(
        role_mapping=dict(user="messages.role.user", assistant="messages.role.assistant")
    ),
    "HuggingFaceH4/ultrachat_200k": dict(
        role_mapping=dict(user="messages.role.user", assistant="messages.role.assistant")
    ),
    "Open-Orca/OpenOrca": dict(role_mapping=dict(system="system_prompt", user="question", assistant="response")),
    "THUDM/LongAlign-10k": dict(role_mapping=dict(user="messages.role.user", assistant="messages.role.assistant")),
    "Yukang/LongAlpaca-12k": dict(role_mapping=dict(user="instruction", assistant="output")),
    "Open-Orca/SlimOrca": dict(
        role_mapping=dict(
            system="conversations.from.system", user="conversations.from.human", assistant="conversations.from.gpt"
        )
    ),
    "meta-math/MetaMathQA": dict(role_mapping=dict(user="query", assistant="response")),
    "ise-uiuc/Magicoder-OSS-Instruct-75K": dict(role_mapping=dict(user="problem", assistant="solution")),
    "lmsys/lmsys-chat-1m": dict(
        role_mapping=dict(user="conversation.role.user", assistant="conversation.role.assistant")
    ),
}


class HFDataSourceConfigInstruct(HFDataSourceConfig):
    role_mapping: Dict[str, str] = Field(default_factory=lambda: {"user": "user", "assistant": "assistant"})
    """
    Flexible mapping from message roles to data extraction paths. Supports:
    - Simple field: {"user": "question", "assistant": "response"}
    - Conversation filter: {"user": "conversations.role.user", "assistant": "conversations.role.assistant"}
    """

    content_key: Optional[str] = Field(default=None)
    """
    The field name to extract content from when using conversation filters.
    If None, will auto-detect using common field names (content, text, message, value).
    Only applies to conversation filter paths (those with dots).
    """

    @field_validator("role_mapping", mode="before")
    @classmethod
    def validate_role_mapping(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Simple validation for role_mapping paths."""
        for role, path_spec in v.items():
            if "." in path_spec:
                # Conversation path: must have exactly 3 parts
                if len(path_spec.split(".")) != 3:
                    raise ValueError(f"Invalid conversation path for role '{role}': expected 'array.field.value'")
        return v

    @model_validator(mode="after")
    def autofill_known_datasets_role_mapping(self) -> Self:
        """Autofill known datasets with default role mappings."""
        dataset_name = str(self.name_or_path).split(":")[0]  # Ignore any split specification
        if dataset_name in KNOWN_HF_INSTRUCT_DATASETS:
            dataset_config = KNOWN_HF_INSTRUCT_DATASETS[dataset_name]
            # Don't override if user provided custom values
            if "role_mapping" not in self.model_fields_set and "role_mapping" in dataset_config:
                role_mapping = dataset_config["role_mapping"]
                self.role_mapping = role_mapping
            if "content_key" not in self.model_fields_set and "content_key" in dataset_config:
                content_key = dataset_config["content_key"]
                self.content_key = content_key
        return self


class HFDataSourceInstruct(HFDataSource):
    """Base DataSource class for instruction-following datasets used in SFT."""

    name = "huggingface_instruct"
    config: HFDataSourceConfigInstruct

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
            messages = self._extract_messages_from_paths(example)

            if not messages:
                # If no messages extracted, try to provide helpful error info
                available_keys = self._get_available_keys(example)
                raise ValueError(
                    f"Could not extract messages using role_mapping: {self.config.role_mapping}. "
                    f"Available data structure: {available_keys}"
                )

            return {"messages": messages}

        # TODO: make it so this special case is not necessary
        if self.config.name_or_path == "nvidia/AceMath-Instruct-Training-Data":
            dataset = dataset.map(
                lambda example: {"messages": example["messages"] + [dict(role="assistant", content=example["answer"])]}
            )

        return dataset.map(
            process_example,
            num_proc=self.data_factory.config.num_proc,
            desc=f"Loading {self.config.name_or_path}",
        )

    def _extract_messages_from_paths(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract messages using flexible path-based mapping."""
        messages: Dict[str, List] = {role: [] for role in self.config.role_mapping.keys()}

        for role, path_spec in self.config.role_mapping.items():
            contents = self._extract_content_from_path(example, path_spec)

            for content in contents:
                if content:
                    messages[role].append({"role": role, "content": content})

        # filter out zero-length
        messages = {role: msgs for role, msgs in messages.items() if len(msgs) > 0}

        # zip and then flatten
        return [msg for msgs in zip(*messages.values()) for msg in msgs]

    def _extract_content_from_path(self, data: Dict[str, Any], path_spec: str) -> List[str]:
        """
        Extract content from data using path specification.

        Supports:
        - Simple field: "question" -> data["question"]
        - Conversation filter: "conversations.role.user" -> find items where role=="user", extract content using content_key
        """
        # Simple field access (no dots)
        if "." not in path_spec:
            value = data.get(path_spec)
            if value is not None:
                return [str(value)]
            return []

        # Parse conversation filter: array_path.filter_field.filter_value
        parts = path_spec.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid conversation path format: {path_spec}. Expected: array_path.filter_field.filter_value"
            )

        array_path = parts[0]  # e.g., "conversations"
        filter_field = parts[1]  # e.g., "role" or "from"
        filter_value = parts[2]  # e.g., "user" or "human"

        # Get the conversation array
        conversations = data.get(array_path)
        if not isinstance(conversations, list):
            return []

        # Find matching items and extract content
        contents = []
        for item in conversations:
            if not isinstance(item, dict):
                continue

            # Check if this item matches our filter
            if item.get(filter_field) == filter_value:
                # Extract content from the matching item
                if self.config.content_key:
                    # Use explicit content key
                    content = item.get(self.config.content_key)
                else:
                    # Use default content field detection
                    content = self._get_default_content(item)

                if content is not None:
                    contents.append(str(content))

        return contents

    def _get_default_content(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract content from an item using common field names."""
        content_candidates = ["content", "text", "message", "value"]
        for candidate in content_candidates:
            if candidate in item and item[candidate] is not None:
                return str(item[candidate])
        return None

    def _get_available_keys(
        self, example: Dict[str, Any], prefix: str = "", max_depth: int = 3
    ) -> Union[Dict[str, Any], str]:
        """Get available keys/structure for error reporting."""
        if max_depth <= 0:
            return "..."

        result: Dict[str, Any] = {}
        for key, value in example.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                result[full_key] = self._get_available_keys(value, full_key, max_depth - 1)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                result[f"{full_key}[0]"] = self._get_available_keys(value[0], f"{full_key}[0]", max_depth - 1)
            else:
                result[full_key] = type(value).__name__

        return result
