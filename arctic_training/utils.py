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

import datetime
import json
import math
from pathlib import Path


def read_json_file(path):
    """read .json or .jsonl file and return its contents"""
    suffix = Path(path).suffix[1:]
    with open(path, "r", encoding="utf-8") as fh:
        if suffix == "jsonl":
            return [json.loads(line) for line in fh]
        elif suffix == "json":
            return json.loads(fh.read())
        else:
            raise ValueError(f"Expected 'json' or 'jsonl' file extension, but got: {suffix}")


def write_json_file(path, data, append=False):
    """write `data` into a .json or .jsonl file"""
    suffix = Path(path).suffix[1:]
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as fh:
        if suffix == "jsonl":
            fh.write(json.dumps(data, sort_keys=True, ensure_ascii=False))
            fh.write("\n")
        elif suffix == "json":
            fh.write(json.dumps(data, sort_keys=True, indent=1, ensure_ascii=False))
        else:
            raise ValueError(f"Expected 'json' or 'jsonl' file extension, but got: {suffix}")


def append_json_file(path, data):
    """append `data` into a .json or .jsonl file"""
    write_json_file(path, data, append=True)


def human_format_base2_number(num: float, suffix: str = "") -> str:
    if num == 0:
        return f"0{suffix}"

    units = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"]
    exponent = min(int(math.log(abs(num), 1024)), len(units) - 1)
    value = num / (1024**exponent)

    return f"{value:_.2f}{units[exponent]}{suffix}"


def human_format_base10_number(num: float, suffix: str = "") -> str:
    if num == 0:
        return f"0{suffix}"

    units = ["", "K", "M", "B", "T", "Qa", "Qi"]  # Qa: Quadrillion, Qi: Quintillion
    exponent = min(int(math.log(abs(num), 1000)), len(units) - 1)
    value = num / (1000**exponent)

    return f"{value:_.2f}{units[exponent]}{suffix}"


def human_format_secs(secs):
    """
    - less than a minute format into seconds with decimals: "%s.%msec"
    - one minute and over use "%H:%M:%S" format
    - if over one day use: "X days, %H:%M:%S" format
    """
    if secs < 60:
        return f"{secs:.3f}s"
    else:
        return str(datetime.timedelta(seconds=secs)).split(".")[0]
