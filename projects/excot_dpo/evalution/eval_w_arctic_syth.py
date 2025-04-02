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
import logging
import os
from typing import Dict

from datasets import Dataset
from execute_utils import _extract_sql
from execute_utils import construct_dnc_prompt
from sql_exec import EvalConfig
from sql_exec import SqlTask
from sql_exec import get_db_path
from sql_exec import load_bird_dataset
from sql_exec import load_spider_dataset
from tqdm import tqdm
from vllm import SamplingParams

from arctic_training.synth import VllmSynth

# Configure logging (if not already configured elsewhere in your project)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class OSSEvalVLLM:
    def __init__(
        self,
        oai_key: str = "DUMMY",
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_tokens: int = 12288,
        task_name: str = None,
        n: int = 1,
        tensor_parallel_size: int = 8,
        prompt_version: str = "divide-and-conquer",
        work_dir: str = "./syth_data",
    ):
        if oai_key:
            self.OAI_key = oai_key
        else:
            self.OAI_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.task_name = task_name
        self.prompt_version = prompt_version

        model_params = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "enable_chunked_prefill": False,
            "download_dir": "/data-fast/vllm",
            "gpu_memory_utilization": 0.95,
            "max_model_len": max_tokens,
            "trust_remote_code": True,
            "swap_space": 64,
        }

        self.client = VllmSynth(
            model_params=model_params,
            sampling_params=SamplingParams(temperature=0.0, n=n, max_tokens=max_tokens),
            work_dir=work_dir,
        )

    def construct_prompt(self, question, db_desc_str):
        db_id = question["db_id"]
        db_desc = db_desc_str[db_id]

        if self.prompt_version == "divide-and-conquer":
            messages = construct_dnc_prompt(db_desc, question["question"], question.get("evidence", None))
        else:
            raise NotImplementedError(f"{self.prompt_version} prompt is not implemented yet.")

        if "difficulty" in question:
            difficulty_levels = question["difficulty"]
        else:
            difficulty_levels = "No levels"
        return messages, difficulty_levels, db_id

    def process_input(self, example, db_desc_str):
        messages, difficulty_levels, db_id = self.construct_prompt(example, db_desc_str)
        messages = [messages[0], messages[-1]]
        return {"messages": messages, "difficulty": difficulty_levels, "db_id": db_id}

    def batch_dataset(self, questions: Dataset, db_desc_str: Dict):
        dataset = questions.map(lambda example: self.process_input(example, db_desc_str))
        messages = dataset["messages"]
        output_data = []

        for idx in range(0, len(messages)):
            row = dataset[idx]
            output_data.append(
                {
                    "custom_id": idx,
                    "db_id": row["db_id"],
                    "difficulty": row["difficulty"],
                }
            )
            self.client.add_chat_to_batch_task(task_name=self.task_name, messages=messages[idx])

        output_data_raw = self.client.extract_messages_from_responses(self.client.execute_batch_task(self.task_name))
        for out_i in output_data_raw:
            custom_id = int(out_i["custom_id"].split("_")[-1])
            output_data[custom_id]["model_output"] = out_i["choices"][0]["content"]

        return output_data


def eval_bird(
    model_name,
    prompt_version,
    eval_config,
    task_name,
    cache_dir="/data-fast/gen_tmp",
    set_type="dev",
    max_len=16384,
    work_dir="./syth_data",
):

    db_desc_str, questions, db_folder = load_bird_dataset(eval_config, set_type, cache_dir)
    cot_generator_vllm = OSSEvalVLLM(
        model_name=model_name,
        task_name=task_name,
        max_tokens=max_len,
        n=1,
        tensor_parallel_size=8,
        prompt_version=prompt_version,
        work_dir=work_dir,
    )
    bird_inferences = cot_generator_vllm.batch_dataset(questions, db_desc_str)
    eval_results = []
    for bird_inference in tqdm(bird_inferences):
        custom_id = bird_inference["custom_id"]
        model_inference = bird_inference["model_output"]
        extracted_sql = _extract_sql(model_inference)
        ground_truth = questions[int(custom_id)]["SQL"]
        db_id = bird_inference["db_id"]
        db_desc = db_desc_str[db_id]
        db_path = get_db_path(db_folder, db_id)

        task = SqlTask(db_id=db_id, ground_truth=ground_truth, db_desc=db_desc, db_path=db_path)
        extracted_sql = _extract_sql(model_inference)
        if extracted_sql:
            extracted_sql = extracted_sql.strip()
        else:
            generation = {
                "question_id": custom_id,
                "correctness": False,
                "answer": model_inference,
                "extracted_sql": None,
                "executed_sql": None,
                "golden_return": ground_truth,
                "difficulty": bird_inference["difficulty"],
            }
            eval_results.append(generation)
            continue

        task.launch_env()
        result = task.exec_sql(extracted_sql)
        task.close_env()

        correctness = set(result) == set(task.answer)
        generation = {
            "question_id": custom_id,
            "correctness": correctness,
            "answer": model_inference,
            "extracted_sql": extracted_sql,
            "executed_sql": result,
            "golden_return": ground_truth,
            "difficulty": bird_inference["difficulty"],
        }
        eval_results.append(generation)

    return eval_results


def eval_spider(
    model_name,
    prompt_version,
    eval_config,
    task_name,
    cache_dir="/data-fast/gen_tmp",
    set_type="test",
    max_len=16384,
):

    db_desc_str, questions, db_folder = load_spider_dataset(eval_config, set_type, cache_dir)
    if not isinstance(questions, Dataset):
        questions = Dataset.from_list(questions)
    cot_generator_vllm = OSSEvalVLLM(
        model_name=model_name,
        task_name=task_name,
        max_tokens=max_len,
        n=1,
        tensor_parallel_size=8,
        prompt_version=prompt_version,
    )
    spider_inferences = cot_generator_vllm.batch_dataset(questions, db_desc_str)

    eval_results = []
    for spider_inference in tqdm(spider_inferences):
        custom_id = spider_inference["custom_id"]
        model_inference = spider_inference["model_output"]
        extracted_sql = _extract_sql(model_inference)
        ground_truth = questions[int(custom_id)]["SQL"]
        db_id = spider_inference["db_id"]
        db_desc = db_desc_str[db_id]
        db_path = get_db_path(db_folder, db_id)
        task = SqlTask(db_id=db_id, ground_truth=ground_truth, db_desc=db_desc, db_path=db_path)
        if extracted_sql:
            extracted_sql = extracted_sql.strip()
        else:
            generation = {
                "question_id": custom_id,
                "correctness": False,
                "answer": model_inference,
                "extracted_sql": None,
                "executed_sql": None,
                "golden_return": ground_truth,
                "difficulty": spider_inference["difficulty"],
            }
            eval_results.append(generation)
            continue

        task.launch_env()
        result = task.exec_sql(extracted_sql)
        correctness = set(result) == set(task.answer)
        task.close_env()
        generation = {
            "question_id": custom_id,
            "correctness": correctness,
            "answer": model_inference,
            "extracted_sql": extracted_sql,
            "executed_sql": result,
            "golden_return": ground_truth,
            "difficulty": spider_inference["difficulty"],
        }
        eval_results.append(generation)

    return eval_results


def calculate_metric_bird(eval_results):
    simple_eval_results = []
    moderate_eval_results = []
    challenging_eval_results = []
    num_of_null = 0
    for eval_res in tqdm(eval_results):
        if eval_res is None:
            num_of_null += 1
            continue
        if eval_res["difficulty"] == "simple":
            simple_eval_results.append(eval_res)
        elif eval_res["difficulty"] == "moderate":
            moderate_eval_results.append(eval_res)
        elif eval_res["difficulty"] == "challenging":
            challenging_eval_results.append(eval_res)
    simple_acc = sum([1 for eval_res in simple_eval_results if eval_res["correctness"]]) / len(simple_eval_results)
    moderate_acc = sum([1 for eval_res in moderate_eval_results if eval_res["correctness"]]) / len(
        moderate_eval_results
    )
    challenging_acc = sum([1 for eval_res in challenging_eval_results if eval_res["correctness"]]) / len(
        challenging_eval_results
    )

    correct_num = 0
    for eval_rest in eval_results:
        if eval_rest is None:
            continue
        if eval_rest["correctness"]:
            correct_num += 1

    exec_acc = correct_num / len(eval_results)
    unexec_num = 0
    for eval_result in eval_results:
        if eval_result is None:
            unexec_num += 1
            continue
        if not ("executed_sql" in eval_result):
            unexec_num += 1
            continue
        if eval_result["executed_sql"] is None:
            unexec_num += 1
            continue
        if (
            ("executed_sql" in eval_result)
            and len(eval_result["executed_sql"]) == 1
            and isinstance(eval_result["executed_sql"][0], str)
        ):
            unexec_num += 1
    exec_succ_rate = (len(eval_results) - unexec_num) / len(eval_results)

    result_dict = {
        "simple_acc": simple_acc,
        "moderate_acc": moderate_acc,
        "challenging_acc": challenging_acc,
        "exec_acc": exec_acc,
        "exec_succ_rate": exec_succ_rate,
        "eval_results": eval_results,
    }

    return result_dict


def calculate_metric_spider(eval_results):

    correct_num = 0
    for eval_rest in eval_results:
        if eval_rest is None:
            continue
        if eval_rest["correctness"]:
            correct_num += 1

    exec_acc = correct_num / len(eval_results)
    unexec_num = 0
    for eval_result in eval_results:
        if eval_result is None:
            unexec_num += 1
            continue
        if not ("executed_sql" in eval_result):
            unexec_num += 1
            continue
        if eval_result["executed_sql"] is None:
            unexec_num += 1
            continue
        if (len(eval_result["executed_sql"]) == 1) and isinstance(eval_result["executed_sql"][0], str):
            unexec_num += 1
    exec_succ_rate = (len(eval_results) - unexec_num) / len(eval_results)
    return exec_acc, exec_succ_rate


def main():
    parser = argparse.ArgumentParser(description="verify the gpt result")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument(
        "--prompt-version",
        type=str,
    )
    parser.add_argument(
        "--data-config",
        type=str,
    )
    parser.add_argument(
        "--mode",
        type=str,
    )
    parser.add_argument(
        "--task-name",
        type=str,
    )
    args = parser.parse_args()

    data_config = EvalConfig()
    data_config.load_yaml(args.data_config)

    if data_config.task == "bird":
        eval_results = eval_bird(
            args.model_name,
            args.prompt_version,
            data_config,
            task_name=args.task_name,
            cache_dir=data_config.cache_dir,
            set_type=args.mode,
        )
        bird_result_dict = calculate_metric_bird(eval_results)
        # Log the results using the logger
        logger.info(
            "simple acc: %s, moderate acc: %s, challenging acc: %s, avg exec acc: %s, avg exec succ rate: %s",
            bird_result_dict["simple_acc"],
            bird_result_dict["moderate_acc"],
            bird_result_dict["challenging_acc"],
            bird_result_dict["exec_acc"],
            bird_result_dict["exec_succ_rate"],
        )
    elif data_config.task == "spider":
        eval_results = eval_spider(
            args.model_name,
            args.prompt_version,
            data_config,
            task_name=args.task_name,
            set_type=args.mode,
        )
        exec_acc, exec_succ_rate = calculate_metric_spider(eval_results)
        logger.info("avg exec acc: %s, avg exec succ rate: %s", exec_acc, exec_succ_rate)


if __name__ == "__main__":
    main()
