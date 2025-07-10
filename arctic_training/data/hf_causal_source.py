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

from arctic_training.data.hf_source import HFDataSource
from arctic_training.data.hf_source import HFDataSourceConfig
from arctic_training.data.utils import DatasetType


class HFDataSourceConfigCausal(HFDataSourceConfig):
    column: str = "text"
    """Column to use for text content extraction."""


class HFDataSourceCausal(HFDataSource):
    """Base DataSource class for causal datasets used in causal training."""

    name = "huggingface_causal"
    config: HFDataSourceConfigCausal

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        if self.config.column not in dataset.column_names:
            if len(dataset.column_names) == 1:
                # If only one column, assume it's the text column
                self.config.column = dataset.column_names[0]
            else:
                raise ValueError(
                    f"Column '{self.config.column}' not found in dataset {self.config.name_or_path}. "
                    "Please provide a valid column name in the `column` field. "
                    f"Available columns: {dataset.column_names}"
                )

        return dataset.map(
            lambda example: dict(text=example[self.config.column]),
            num_proc=self.data_factory.config.num_proc,
            desc=f"Loading causal dataset {self.config.name_or_path}",
        )
