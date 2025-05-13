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
from typing import Set
from typing import Tuple

import torch
from pydantic_core import core_schema


class DType(Enum):
    _aliases: Set[Any]

    FP16 = torch.float16, "torch.float16", "fp16", "float16", "half"
    FP32 = torch.float32, "torch.float32", "fp32", "float32", "float"
    BF16 = torch.bfloat16, "torch.bfloat16", "bf16", "bfloat16", "bfloat"

    def __new__(cls, value: Any, *aliases: Tuple[Any]):
        obj = object.__new__(cls)
        obj._value_ = value
        obj._aliases = {value, *aliases}
        return obj

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if value in member._aliases:
                return member
        return None

    def __eq__(self, other) -> bool:
        if isinstance(other, DType):
            return self is other
        return other in self._aliases

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, _handler):
        return core_schema.no_info_after_validator_function(
            lambda v: cls(v),
            core_schema.any_schema(),  # Allow str or torch.dtype input
            serialization=core_schema.plain_serializer_function_ser_schema(cls.serialize),
        )

    @classmethod
    def serialize(cls, instance):
        return str(instance.value)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string", "title": "Torch Dtype", "examples": ["float32", "bfloat16"]}
