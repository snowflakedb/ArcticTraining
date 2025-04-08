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

import argparse

from data_generation import construct_gpt_prompt
from datasets import Dataset
from datasets import load_from_disk


def main():
    parser = argparse.ArgumentParser(description="verify the gpt result")
    parser.add_argument("--verify-path", help="Chain of thought file path.", type=str)
    parser.add_argument(
        "--output-path",
        type=str,
    )
    args = parser.parse_args()

    data_verify = load_from_disk(args.verify_path)

    new_data = []
    for row in data_verify:
        if len(row["correct_answers"]) > 0:
            new_messages = construct_gpt_prompt(row)
            if len(new_messages) > 2:
                new_messages = [new_messages[0], new_messages[-1]]
            new_messages.append({"role": "assistant", "content": row["correct_answers"][0]})
            new_data.append(new_messages)

    new_dataset = Dataset.from_dict(
        {
            "messages": [d for d in new_data],
        }
    )
    new_dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
