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

import math


def human_format_base2_number(num: float, suffix: str = "") -> str:
    if num == 0:
        return f"0{suffix}"

    units = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"]
    exponent = min(int(math.log(abs(num), 1024)), len(units) - 1)
    value = num / (1024**exponent)

    return f"{value:_.1f}{units[exponent]}{suffix}"


def human_format_base10_number(num: float, suffix: str = "") -> str:
    if num == 0:
        return f"0{suffix}"

    units = ["", "K", "M", "B", "T", "Qa", "Qi"]  # Qa: Quadrillion, Qi: Quintillion
    exponent = min(int(math.log(abs(num), 1000)), len(units) - 1)
    value = num / (1000**exponent)

    return f"{value:_.1f}{units[exponent]}{suffix}"
