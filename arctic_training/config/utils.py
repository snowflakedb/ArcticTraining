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

import os
import re
from typing import Annotated
from typing import Union

import yaml
from pydantic.functional_validators import BeforeValidator


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", 0))


def get_global_rank() -> int:
    return int(os.getenv("RANK", 0))


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", 1))


# From https://gist.github.com/pypt/94d747fe5180851196eb?permalink_comment_id=4015118#gistcomment-4015118
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate '{key}' key found in YAML on line {key_node.start_mark.line + 1}.")
            mapping.add(key)
        return super().construct_mapping(node, deep)


def parse_human_val(value: Union[str, int, float]) -> float:
    if isinstance(value, str):
        value = value.replace("_", "")
        value = value.lower().strip()

        # Handle percentage values
        if value.endswith("%"):
            return float(value[:-1]) / 100

        # Handle X^Y expressions
        match_exp = re.match(r"^(-?\d+\.?\d?)\^(-?\d+\.?\d?)$", value)
        if match_exp:
            base, exp = map(float, match_exp.groups())
            sign = -1 if base < 0 else 1
            return sign * abs(base) ** exp

        # Handle XeY expressions
        match_exp = re.match(r"^(-?\d+\.?\d?)e(-?\d+\.?\d?)$", value)
        if match_exp:
            base, exp = map(float, match_exp.groups())
            return base * 10**exp

        # Handle suffixes like k, m, b (base 10) and ki, mi, gi (base 2)
        suffixes = {s: 10 ** (i * 3) for i, s in enumerate(("k", "m", "b", "t"), start=1)}
        suffixes.update({s: 2 ** (i * 10) for i, s in enumerate(("ki", "mi", "gi", "ti"), start=1)})
        match_suffix = re.match(rf"^(-?\d+\.?\d?)({'|'.join(suffixes.keys())})$", value)
        if match_suffix:
            num, suffix = match_suffix.groups()
            return float(num) * suffixes[suffix]

    # Fallback to python conversion
    return float(value)


# Note: parse_human_val will return a float, but pydantic will handle the conversion to int if the field is an int
HumanInt = Annotated[int, BeforeValidator(parse_human_val)]
HumanFloat = Annotated[float, BeforeValidator(parse_human_val)]
