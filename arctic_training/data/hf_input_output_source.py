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

from pydantic import Field

from arctic_training.data.hf_source import HFDataSource
from arctic_training.data.hf_source import HFDataSourceConfig
from arctic_training.data.utils import DatasetType


class HFDataSourceConfigInputOutput(HFDataSourceConfig):
    input_key: str = Field(default="input")
    """The field name for the input/instruction part of the data."""

    output_key: str = Field(default="output")
    """The field name for the output/response part of the data."""


class HFDataSourceInputOutput(HFDataSource):
    """
    DataSource class for datasets with INPUT/OUTPUT format.
    
    This source handles datasets where:
    - Data contains separate INPUT and OUTPUT fields
    - The full sequence is INPUT + OUTPUT
    - Only the OUTPUT portion is used for training (INPUT is masked)
    """

    name = "huggingface_input_output"
    config: HFDataSourceConfigInputOutput

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
            input_text = example.get(self.config.input_key, "")
            output_text = example.get(self.config.output_key, "")
            
            if not output_text:
                raise ValueError(
                    f"Output field '{self.config.output_key}' is empty or missing. "
                    f"Available keys: {list(example.keys())}"
                )
            
            return {
                "input": str(input_text),
                "output": str(output_text),
            }

        return dataset.map(
            process_example,
            num_proc=self.data_factory.config.num_proc,
            desc=f"Loading {self.config.name_or_path}",
        )
