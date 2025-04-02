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

import editdistance
from datasets import Dataset
from datasets import load_from_disk
from prompts.divide_and_conquer import messages as dnc_messages
from tqdm import tqdm


def create_conv_v15(schema, question, chosen, reject):
    prompt = f"""
Database Info
{schema}
**************************
Question
Question: {question}
**************************
""".strip()
    prompt_message = [dnc_messages[0], {"content": prompt, "role": "user"}]
    chosen_message = [{"content": chosen, "role": "assistant"}]
    rejected_message = [{"content": reject, "role": "assistant"}]
    return {
        "prompt": prompt_message,
        "chosen": chosen_message,
        "rejected": rejected_message,
    }


def get_closest_match(query, candidates):
    min_distance = 10000
    closest = None
    closest_idx = None
    for i, candidate in enumerate(candidates):
        distance = editdistance.eval(query, candidate)
        if distance < min_distance:
            min_distance = distance
            closest = candidate
            closest_idx = i
    return closest_idx, closest, min_distance


def get_max_match(query, candidates):
    max_distance = -1
    furthest = None
    furthest_idx = None
    for i, candidate in enumerate(candidates):
        distance = editdistance.eval(query, candidate)
        if distance > max_distance:
            max_distance = distance
            furthest = candidate
            furthest_idx = i
    return furthest_idx, furthest, max_distance


def sample_pairs_v1(data):
    data_pairs = []
    for row in tqdm(data):
        schema = row["schema"]
        question = row["question"]
        if ("evidence" in row) and not (row["evidence"] is None):
            question = question + " " + row["evidence"]
        num_pairs = min(1, len(row["correct_answers"]), len(row["wrong_answers"]))
        if num_pairs == 0:
            continue
        for i in range(num_pairs):
            chosen = row["correct_answers"][i]
            reject = row["wrong_answers"][i]
            if chosen is None or reject is None:
                continue
            data_pairs.append((schema, question, chosen, reject))
    return data_pairs


def sample_pairs_v2_1(data):
    data_pairs = []
    for row in tqdm(data):
        schema = row["schema"]
        question = row["question"]
        if ("evidence" in row) and not (row["evidence"] is None):
            question = question + " " + row["evidence"]
        num_pairs = min(len(row["correct_answers"]), len(row["wrong_answers"]))
        if num_pairs == 0:
            continue

        current_min_dist = 1000000
        current_min_pair = None
        for i, chosen in enumerate(row["correct_answers"]):
            reject_idx, reject, min_dist = get_closest_match(chosen, row["wrong_answers"])
            if min_dist < current_min_dist:
                current_min_dist = min_dist
                current_min_pair = (chosen, reject)

        data_pairs.append((schema, question, current_min_pair[0], current_min_pair[1]))
    return data_pairs


def sample_pairs_v2_2(data):
    data_pairs = []
    for row in tqdm(data):
        schema = row["schema"]
        question = row["question"]
        if ("evidence" in row) and not (row["evidence"] is None):
            question = question + " " + row["evidence"]
        num_pairs = min(len(row["correct_answers"]), len(row["wrong_answers"]))
        if num_pairs == 0:
            continue

        current_max_dist = -1
        current_max_pair = None
        for i, chosen in enumerate(row["correct_answers"]):
            reject_idx, reject, max_dist = get_max_match(chosen, row["wrong_answers"])
            if max_dist > current_max_dist:
                current_max_dist = max_dist
                current_max_pair = (chosen, reject)

        data_pairs.append((schema, question, current_max_pair[0], current_max_pair[1]))
    return data_pairs


def process_data(data, version):
    dpo_data = []
    if version == "random":
        data_pairs = sample_pairs_v1(data)
    elif version == "nearest":
        data_pairs = sample_pairs_v2_1(data)
    elif version == "farthest":
        data_pairs = sample_pairs_v2_2(data)
    for schema, question, chosen, reject in data_pairs:
        dpo_data.append(create_conv_v15(schema, question, chosen, reject))
    return dpo_data


def main():
    parser = argparse.ArgumentParser(description="Your program description.")
    parser.add_argument(
        "--dataset-path",
        type=str,
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["random", "nearest", "farthest"],
    )
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()
    cot_processed_data = load_from_disk(args.dataset_path)
    dpo_data = process_data(cot_processed_data, args.version)
    new_dataset = Dataset.from_dict(
        {
            "prompt": [d["prompt"] for d in dpo_data],
            "chosen": [d["chosen"] for d in dpo_data],
            "rejected": [d["rejected"] for d in dpo_data],
        }
    )
    new_dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()
