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

from arctic_training.data.hf_source import HFDataSource
from arctic_training.data.hf_source import HFDataSourceConfig
from arctic_training.data.utils import DatasetType


class HFDataSourceConfigCausal(HFDataSourceConfig):
    column_mapping: Dict[str, str] = Field(default_factory=lambda: {"text": "text"})
    """
    Flexible mapping from content key to data extraction paths. Supports:
    - Simple field: {"text": "text"}
    - Conversation filter: {"text": "conversations.role.user"}
    If not provided and only one column is present in the dataset, that column will be used as the text column.
    """

    content_key: Optional[str] = Field(default=None)
    """
    The field name to extract content from when using column_mapping.
    If None, will auto-detect using common field names (content, text, message, value).
    Only applies to column_mapping paths (those with dots).
    """

    template: str = "{text}"
    """
    Template used to format the columns of each data sample.
    """

    @field_validator("column_mapping", mode="before")
    @classmethod
    def validate_column_mapping(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Simple validation for column_mapping paths."""
        for column, path_spec in v.items():
            if "." in path_spec:
                # Conversation path: must have exactly 3 parts
                if len(path_spec.split(".")) != 3:
                    raise ValueError(f"Invalid conversation path for column '{column}': expected 'array.field.value'")
        return v


class HFDataSourceCausal(HFDataSource):
    """Base DataSource class for causal datasets used in causal training."""

    name = "huggingface_causal"
    config: HFDataSourceConfigCausal

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
            content_mapping = self._extract_content_from_mapping(example, self.config.column_mapping)

            if not content_mapping:
                # If no content extracted, try to provide helpful error info
                available_keys = self._get_available_keys(example)
                raise ValueError(
                    f"Could not extract content using column_mapping: {self.config.column_mapping}. "
                    f"Available data structure: {available_keys}"
                )

            # Apply template formatting
            formatted_content = self.config.template.format(**content_mapping)
            return {"text": formatted_content}

        if "column_mapping" not in self.config.model_fields_set and "text" not in dataset.column_names:
            if len(dataset.column_names) == 1:
                self.config.column_mapping = {"text": dataset.column_names[0]}
            else:
                raise ValueError(
                    f"Could not auto-detect column mapping for dataset {self.config.name_or_path}. Please provide a"
                    f" column_mapping. Available columns: {dataset.column_names}"
                )

        return dataset.map(
            process_example,
            num_proc=self.data_factory.config.num_proc,
            desc=f"Loading causal dataset {self.config.name_or_path}",
        )

    def _extract_content_from_mapping(self, example: Dict[str, Any], column_mapping: Dict[str, str]) -> Dict[str, str]:
        """Extract content using flexible path-based mapping."""
        content_mapping = {}

        for column, path_spec in column_mapping.items():
            contents = self._extract_content_from_path(example, path_spec)

            if contents:
                # For causal datasets, we typically want the first/main content
                content_mapping[column] = contents[0]

        return content_mapping

    def _extract_content_from_path(self, data: Dict[str, Any], path_spec: str) -> List[str]:
        """
        Extract content from data using path specification.

        Supports:
        - Simple field: "text" -> data["text"]
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
