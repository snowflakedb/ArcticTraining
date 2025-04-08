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
import json
import os
from multiprocessing import Pool
from multiprocessing import TimeoutError
from typing import Dict
from typing import List

from data_generation import DataGenerationConfig
from data_generation import dataset_conversion
from data_generation import load_bird_dataset
from data_generation import load_spider_dataset
from datasets import Dataset
from datasets import load_from_disk
from tqdm import tqdm
from utils.execute_utils import _extract_sql
from utils.sql_exec import SqlTask
from utils.sql_exec import get_db_path


def read_jsonl(jsonl_path: str):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_json_file(json_path: str):
    """
    Loads and returns the contents of a JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"The file {json_path} does not exist.")

    with open(json_path, "r") as file:
        data = json.load(file)
    return data


def read_reuslts(data_result: List[Dict]):
    result_dict = {}
    for result in data_result:
        custom_id = int(result["custom_id"].split("_")[-1])
        if result["error"] is None:
            try:
                choices = result["response"]["body"]["choices"]  # [0]['message']['content']
                for choice in choices:
                    text = choice["message"]["content"]
                    if custom_id not in result_dict:
                        result_dict[custom_id] = {}
                        result_dict[custom_id]["all_generated_queries"] = [text]
                    else:
                        result_dict[custom_id]["all_generated_queries"].append(text)
            except KeyError:
                continue
    print(f"Length of result_dict: {len(result_dict)}")
    return result_dict


def verify_one_line(check_row, bird_source, db_folder):
    schema = check_row["schema"]
    question = check_row["question"]
    golden_answer = check_row["golden_query"]
    custom_id = check_row["custom_id"]

    db_id = bird_source[custom_id]["db_id"]
    assert bird_source[custom_id]["question"] == question
    db_path = get_db_path(db_folder, db_id)

    task = SqlTask(db_id=db_id, ground_truth=golden_answer, db_desc=schema, db_path=db_path)
    task.launch_env()
    for answer in check_row["all_generated_queries"]:
        extracted_sql = _extract_sql(answer)
        if extracted_sql is None:
            generation = {"question_id": custom_id, "correctness": False, "answer": ""}

        else:
            if set(task.answer) is None or (set(task.answer) == set({})):
                correctness = False
            else:
                result = task.exec_sql(extracted_sql)
                correctness = set(result) == set(task.answer)
            generation = {
                "question_id": custom_id,
                "correctness": correctness,
                "answer": answer,
                "extracted_sql": extracted_sql,
            }

        if generation["correctness"]:
            check_row["correct_answers"].append(answer)
        else:
            check_row["wrong_answers"].append(answer)
    return check_row


def wrapper(args):
    return verify_one_line(*args)


def check_correctness(checking_list, db_desc_str, db_folder, bird_source, timeout=5):

    num_process = 1

    eval_results = []
    chunk_size = 32
    if num_process > 1:
        for cid in tqdm(
            range(0, len(checking_list), chunk_size),
            total=len(checking_list) // chunk_size,
        ):
            check_sub_list = checking_list[cid : cid + chunk_size]
            with Pool(num_process) as p:
                async_results = [
                    p.apply_async(wrapper, ((check_row, bird_source, db_folder),)) for check_row in check_sub_list
                ]

                for async_result in async_results:
                    try:
                        result = async_result.get(timeout=timeout)
                        eval_results.append(result)
                    except TimeoutError:
                        # Handle the timeout case
                        eval_results.append(None)  # or any other default value
                    except Exception:
                        # Handle any other exceptions
                        eval_results.append(None)  # or handle differently as needed
    else:
        for check_row in checking_list:
            eval_results.append(wrapper((check_row, bird_source, db_folder)))

    final_result = [None] * len(bird_source)
    for result in eval_results:
        if result is not None:
            result_id = result["custom_id"]
            final_result[result_id] = result

    final_result = [d for d in final_result if d is not None]
    return final_result


class Verifier:
    def __init__(self, data_config_path: str):
        self.data_config = DataGenerationConfig()
        self.data_config.load_yaml(data_config_path)
        self.cache_dir = self.data_config.cache_dir
        self.source_data = dataset_conversion(self.data_config)

    def verify_dataset(
        self,
        gpt_path: str,
        vllm_path: str,
        output_path: str,
    ):
        if self.data_config.task == "bird":
            db_desc_str, questions, db_folder = load_bird_dataset(self.data_config, "train", self.cache_dir)

        elif self.data_config.task == "spider":
            db_desc_str, questions, db_folder = load_spider_dataset(self.data_config, "train", self.cache_dir)

        if gpt_path is not None:
            gpt_json = read_jsonl(gpt_path)
            generated_results = read_reuslts(gpt_json)
        elif vllm_path is not None:
            generated_dataset = load_from_disk(vllm_path)
            generated_results = {}
            for i, row in enumerate(generated_dataset):
                assert row["custom_id"] == i
                generated_results[i] = {}
                generated_results[i]["all_generated_queries"] = row["all_generated_queries"]
        else:
            # This branch should never be reached because of the assert above.
            assert False, "No valid path provided."

        full_data = []

        for i, f_data in enumerate(self.source_data):
            if "custom_id" in f_data:
                custom_id = f_data["custom_id"]
            else:
                custom_id = i
            if i not in generated_results:
                continue

            generated_result = generated_results[i]
            row = self.source_data[custom_id]
            question = row["question"]
            schema = row["schema"]
            if "evidence" in f_data:
                evidence = f_data["evidence"]
            else:
                evidence = None
            golden_answer = row["expected_answer"]
            all_generated_queries = generated_result["all_generated_queries"]

            full_data.append(
                {
                    "custom_id": custom_id,
                    "schema": schema,
                    "question": question,
                    "evidence": evidence,
                    "all_generated_queries": all_generated_queries,
                    "correct_answers": [],
                    "wrong_answers": [],
                    "golden_query": golden_answer,
                }
            )

        processed_data = check_correctness(full_data, db_desc_str, db_folder, questions)

        dataset = {
            "custom_id": [data["custom_id"] for data in processed_data],
            "schema": [data["schema"] for data in processed_data],
            "question": [data["question"] for data in processed_data],
            "golden_query": [data["golden_query"] for data in processed_data],
            "correct_answers": [data["correct_answers"] for data in processed_data],
            "wrong_answers": [data["wrong_answers"] for data in processed_data],
        }

        if "evidence" in processed_data[0]:
            dataset["evidence"] = [data["evidence"] for data in processed_data]

        dataset = Dataset.from_dict(dataset)
        dataset.save_to_disk(output_path)


def main():
    parser = argparse.ArgumentParser(description="verify the gpt result")
    parser.add_argument(
        "--config-path",
        type=str,
    )
    parser.add_argument("--gpt-cot-path", help="Chain of thought file path.", type=str, default=None)
    parser.add_argument("--vllm-cot-path", help="Chain of thought file path.", type=str, default=None)
    parser.add_argument(
        "--output-path",
        help="original dataset path",
        type=str,
        default="/data-fast/test_arctic",
    )

    args = parser.parse_args()
    verifier = Verifier(args.config_path)
    verifier.verify_dataset(
        args.gpt_cot_path,
        args.vllm_cot_path,
        args.output_path,
    )


if __name__ == "__main__":
    main()
