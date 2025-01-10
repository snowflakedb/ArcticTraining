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

from enum import Enum
from typing import Any
from typing import Tuple

import torch


class DType(Enum):
    _all_values: Tuple[Any]  # To make mypy happy

    # The torch dtype must always be the first value (so we return torch.dtype)
    FP16 = torch.float16, "torch.float16", "fp16", "float16", "half"
    FP32 = torch.float32, "torch.float32", "fp32", "float32", "float"
    BF16 = torch.bfloat16, "torch.bfloat16", "bf16", "bfloat16", "bfloat"
    INT8 = torch.int8, "torch.int8", "int8"

    # Copied from https://stackoverflow.com/a/43210118
    # Allows us to use multiple values for each Enum index and returns first
    # listed value when Enum is called
    def __new__(cls, *values: Any) -> "DType":
        obj = object.__new__(cls)
        # first value is canonical value
        obj._value_ = values[0]
        for other_value in values[1:]:
            cls._value2member_map_[other_value] = obj
        obj._all_values = values
        return obj

    def __repr__(self) -> str:
        return "<%s.%s: %s>" % (
            self.__class__.__name__,
            self._name_,
            ", ".join([repr(v) for v in self._all_values]),
        )
