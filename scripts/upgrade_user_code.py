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

import re
import sys
from pathlib import Path


def update_file(file_path: Path):
    with open(file_path, "r") as f:
        content = f.read()

    union_import_needed = False

    class_attrs = [
        "config_type",
        "data_factory_type",
        "model_factory_type",
        "checkpoint_engine_type",
        "optimizer_factory_type",
        "scheduler_factory_type",
        "tokenizer_factory_type",
    ]

    for attr in class_attrs:
        content = re.sub(
            rf"{attr}\s*=\s*([A-Za-z_][A-Za-z0-9_]*)",
            lambda match, attr_name=attr.replace("_type", ""): f"{attr_name}: {match.group(1)}",
            content,
        )

    def replace_with_union(match, attr_name):
        nonlocal union_import_needed
        union_import_needed = True
        return f"{attr_name}: Union[{match.group(1)}]"

    for attr in class_attrs:
        content = re.sub(
            rf"{attr}\s*=\s*\[([A-Za-z0-9_,\s]+)\]",
            lambda match, attr_name=attr.replace("_type", ""): replace_with_union(match, attr_name),
            content,
        )

    if union_import_needed and not re.search(r"from typing import .*Union.*", content):
        content = "from typing import Union\n" + content

    with open(file_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python upgrade_user_code.py <user-code.py>")
        sys.exit(1)
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File {file_path} not found")
        sys.exit(1)
    update_file(file_path)
