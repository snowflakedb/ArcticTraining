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
from typing import Dict
from typing import List

from datasets import DatasetDict
from datasets import load_dataset
from datasets import load_from_disk
from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from arctic_training.config.data import DataSourceConfig
from arctic_training.data.source import DataSource
from arctic_training.data.utils import DatasetType

# Known datasets with split mappings
KNOWN_HF_DATASETS: Dict[str, Dict[str, Dict[str, str]]] = {
    "HuggingFaceH4/ultrachat_200k": dict(split_mapping=dict(train="train_sft", eval="test_sft")),
    "HuggingFaceH4/ultrafeedback_binarized": dict(split_mapping=dict(train="train_prefs", eval="test_prefs")),
    "nvidia/AceMath-Instruct-Training-Data": dict(
        split_mapping=dict(train="general_sft_stage2"), kwargs=dict(verification_mode="no_checks")
    ),
}


class HFDataSourceConfig(DataSourceConfig):
    name_or_path: str
    """
    Name or path of the dataset to load. Also accepts values for the split field
    after a colon (e.g. "name:split", "name:split[10:20]").
    """

    kwargs: Dict[str, Any] = Field(default_factory=dict)
    """ Keyword arguments to pass to the datasets.load_dataset function. """

    split_mapping: Dict[str, str] = Field(default_factory=dict)
    """
    Mapping from standard split names to dataset-specific split names.
    E.g., {"train": "train_sft", "eval": "test_sft"} for ultrachat_200k.
    """

    @model_validator(mode="after")
    def get_split_from_name_or_path(self) -> Self:
        if ":" in self.name_or_path:
            self.name_or_path, self.split = self.name_or_path.split(":", 1)
        return self

    @model_validator(mode="after")
    def autofill_known_datasets_split_mapping(self) -> Self:
        """Autofill split mappings for known datasets."""
        if (
            self.name_or_path in KNOWN_HF_DATASETS
            and "split_mapping" in KNOWN_HF_DATASETS[self.name_or_path]
            and not self.split_mapping
        ):
            self.split_mapping = KNOWN_HF_DATASETS[self.name_or_path]["split_mapping"]
        return self

    @model_validator(mode="after")
    def autofill_known_datasets_kwargs(self) -> Self:
        """Autofill kwargs for known datasets."""
        if self.name_or_path in KNOWN_HF_DATASETS and "kwargs" in KNOWN_HF_DATASETS[self.name_or_path]:
            for key, value in KNOWN_HF_DATASETS[self.name_or_path]["kwargs"].items():
                if key not in self.kwargs:
                    self.kwargs[key] = value
        return self


class HFDataSource(DataSource):
    """Base DataSource class for loading data with HuggingFace datasets library."""

    name = "huggingface"
    config: HFDataSourceConfig

    def pre_load_callback(self, split: str) -> str:
        """Apply split mapping if configured for this dataset."""
        if self.config.split_mapping:
            # Extract the base split name (before any slice notation like [:5])
            base_split = split.split("[")[0] if "[" in split else split

            # Check if the base split is already a target split (already mapped)
            target_splits = set(self.config.split_mapping.values())
            if base_split in target_splits:
                # Already in target form, no mapping needed
                return split

            # Apply mapping only if we have an exact match for the base split
            if base_split in self.config.split_mapping:
                mapped_split = self.config.split_mapping[base_split]
                # Replace only the base split part, preserve any slice notation
                if "[" in split:
                    slice_part = split[split.index("[") :]
                    split = mapped_split + slice_part
                else:
                    split = mapped_split
        return split

    def load(self, config: HFDataSourceConfig, split: str) -> DatasetType:
        # Support loading local datasets
        if Path(config.name_or_path).exists():
            dataset = load_from_disk(config.name_or_path, **config.kwargs)
            if isinstance(dataset, DatasetDict):
                dataset = dataset[split]
        else:
            dataset = load_dataset(config.name_or_path, split=split, **config.kwargs)

        return dataset


class ProjectGutenbergSFT(HFDataSource):
    """
    Simple SFT wrapper around the Project Gutenberg dataset. Each example only
    contains a single user message with the text content, and no assistant
    response. This is intended for distillation on prompt tokens like SwiftKV.
    """

    name = "ProjectGutenbergSFT"

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:

        def process_example(example):
            return {"messages": [{"role": "user", "content": example["text"]}]}

        return dataset.map(
            process_example,
            num_proc=self.data_factory.config.num_proc,
            desc="Loading Project Gutenberg",
        )


class UltraFeedbackBinarized(HFDataSource):
    name = "HuggingFaceH4/ultrafeedback_binarized"

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        dataset = dataset.select_columns(["chosen", "rejected"])
        formatted_dataset = dataset.map(self.split_prompt_content, desc="Loading ultrafeedback binarized")
        return formatted_dataset

    @staticmethod
    def split_prompt_content(example: Dict[str, List]) -> Dict[str, List]:
        r"""
        Extracts the shared prompt from a preference data example, where the prompt is implicit within both
        the chosen and rejected completions.

        For more details, see [`maybe_extract_prompt`].
        """
        for idx in range(min(len(example["chosen"]), len(example["rejected"]))):
            if example["chosen"][idx]["content"] != example["rejected"][idx]["content"]:
                break
        return {
            "prompt": example["chosen"][:idx],
            "chosen": example["chosen"][idx:],
            "rejected": example["rejected"][idx:],
        }
