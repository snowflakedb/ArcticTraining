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

import hashlib
from typing import Any
from typing import Union

from datasets import Dataset
from datasets import IterableDataset

DatasetType = Union[Dataset, IterableDataset]


def calculate_hash_from_args(*args: Any) -> str:
    hash_str = ""
    for arg in args:
        try:
            hash_str += str(arg)
        except Exception as e:
            raise ValueError(
                f"Failed to convert {arg} to string when calculating cache path."
                f" Error: {e}"
            )
    return hashlib.md5(hash_str.encode()).hexdigest()
