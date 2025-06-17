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
import re

import psutil
import ray
import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams


@ray.remote
class VLLMServer:
    def __init__(self, model_params):
        """Initialize vLLM model."""
        self.llm = LLM(**model_params)
        self.model_params = model_params

        # #
        # state_dict = torch.load(args.model_path + '/pytorch_model.bin', map_location="cpu")
        # llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        # weights = [(name, p) for name, p in state_dict.items()]
        # llm_model.load_weights(weights)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        """Generate response from LLM."""
        result = self.llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
        return result


def partition_list(lst, n):
    """Partition a list into n parts as evenly as possible."""
    avg_size = len(lst) // n
    remainder = len(lst) % n
    partitions = []
    start = 0

    for i in range(n):
        extra = 1 if i < remainder else 0  # Distribute remainder items
        end = start + avg_size + extra
        partitions.append(lst[start:end])
        start = end

    return partitions


def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"

    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        # print("No SQL blocks found.")
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/fs/fast/u2021000902/previous_nvme/xxx")
    parser.add_argument("--input_file", type=str, help="the input file path (prompts)")
    parser.add_argument("--output_file", type=str, help="the output file path (results)")
    parser.add_argument("--tensor_parallel_size", type=int, help="the number of used GPUs", default=4)
    parser.add_argument("--n", type=int, help="the number of generated responses", default=4)
    parser.add_argument("--temperature", type=float, help="temperature of llm's sampling", default=1.0)
    parser.add_argument("--parallel_generation", action="store_true", help="Using ray + vllm to speed up generation.")

    opt = parser.parse_args()
    print(opt)

    if opt.parallel_generation:
        num_gpus = torch.cuda.device_count()
        print(f"AVAILABLE GPU COUNT: {num_gpus}")
        max_cpus = psutil.cpu_count(logical=False)
        print(f"AVAILABLE CPU COUNT: {max_cpus}")
        ray.init(num_cpus=max_cpus, num_gpus=num_gpus, include_dashboard=False, ignore_reinit_error=True)

    input_dataset = json.load(open(opt.input_file))
    tokenizer = AutoTokenizer.from_pretrained(opt.pretrained_model_name_or_path, trust_remote_code=True)

    if "arctic" in opt.pretrained_model_name_or_path.lower():
        stop_token_ids = [151645]
    elif "Qwen2.5-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [151645]  # 151645 is the token id of <|im_end|> (end of turn token in Qwen2.5)
    elif "OmniSQL-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [151645]  # OmniSQL uses the same tokenizer as Qwen2.5
    elif "deepseek-coder-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [32021]
    elif "DeepSeek-Coder-V2" in opt.pretrained_model_name_or_path:
        stop_token_ids = [100001]
    elif "OpenCoder-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [96539]
    elif "Meta-Llama-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [128009, 128001]
    elif "granite-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [0]  # <|end_of_text|> is the end token of granite-3.1 and granite-code
    elif "starcoder2-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [0]  # <|end_of_text|> is the end token of starcoder2
    elif "Codestral-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [2]
    elif "Mixtral-" in opt.pretrained_model_name_or_path:
        stop_token_ids = [2]
    else:
        print("Use Qwen2.5's stop tokens by default.")
        stop_token_ids = [151645]

    print("stop_token_ids:", stop_token_ids)

    max_model_len = 16384  # used to allocate KV cache memory in advance
    max_input_len = 8192
    max_output_len = 8192  # (max_input_len + max_output_len) must <= max_model_len

    print("max_model_len:", max_model_len)
    print("temperature:", opt.temperature)
    sampling_params = SamplingParams(
        temperature=opt.temperature, max_tokens=max_output_len, n=opt.n, stop_token_ids=stop_token_ids
    )

    if opt.parallel_generation:
        if "7b" in opt.pretrained_model_name_or_path:
            vllm_tp_size = 1
        elif "14b" in opt.pretrained_model_name_or_path:
            vllm_tp_size = 2
        elif "32b" in opt.pretrained_model_name_or_path:
            vllm_tp_size = 4
        else:
            vllm_tp_size = 8

        num_vllm_instance = int(8 // vllm_tp_size)
        model_params = {
            "model": opt.pretrained_model_name_or_path,
            "dtype": "bfloat16",
            "tensor_parallel_size": vllm_tp_size,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": 0.92,
            "swap_space": 42,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
            "trust_remote_code": True,
        }

        llm_list = [VLLMServer.options(num_gpus=vllm_tp_size).remote(model_params) for i in range(num_vllm_instance)]
    else:
        llm = LLM(
            model=opt.pretrained_model_name_or_path,
            dtype="bfloat16",
            tensor_parallel_size=opt.tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.92,
            swap_space=42,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
        )

    chat_prompts = []
    for data in input_dataset:
        cot_info = "Let me solve this step by step. \n<think>"
        instruct_info = """
Please provide a detailed chain-of-thought reasoning process and include your thought process within `<think>` tags. Your final answer should be enclosed within `<answer>` tags.

Ensure that your SQL query follows the correct syntax and is formatted as follows:

```sql
-- Your SQL query here
```

Example format:
<think> Step-by-step reasoning, including self-reflection and corrections if necessary. [Limited by 4K tokens] </think>
<answer> Summary of the thought process leading to the final SQL query. [Limited by 1K tokens]

```sql
Correct SQL query here
```
</answer>""".strip()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a data science expert. Below, you are provided with a database schema and a natural"
                    " language question. Your task is to understand the schema and generate a valid SQL query to"
                    " answer the question."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"""
Database Engine:
SQLite

Database Schema:
{data["db_desc"]}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{data["question"]}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
{instruct_info}
    """.strip()
                ),
            },
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt += cot_info
        chat_prompts.append(prompt)
    # import pdb; pdb.set_trace()

    if opt.parallel_generation:
        # split the input prompts into num_vllm_instance parts
        chat_prompts_list = partition_list(chat_prompts, num_vllm_instance)
        outputs = ray.get([llm.generate.remote(chat_prompts, sampling_params) for llm in llm_list])
        outputs = sum(outputs, [])
    else:
        outputs = llm.generate(chat_prompts, sampling_params)

    results = []
    for data, output in zip(input_dataset, outputs):
        responses = [o.text for o in output.outputs]
        sqls = [parse_response(response) for response in responses]

        data["responses"] = responses
        data["pred_sqls"] = sqls
        results.append(data)

    with open(opt.output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
