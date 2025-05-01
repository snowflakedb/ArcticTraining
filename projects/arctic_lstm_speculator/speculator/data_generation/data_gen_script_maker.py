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
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model_name", required=True)
parser.add_argument("--data_save_folder_name", required=True)
parser.add_argument("--vllm_tensor_parallel", type=int, default=1)
parser.add_argument("--script_save_path", required=True)
parser.add_argument("--total_num_of_scripts", type=int, default=8)
args = parser.parse_args()
print(args)

model_name = tokenizer_name = args.model_name
data_save_folder_name = args.data_save_folder_name
script_save_path = args.script_save_path
os.makedirs(script_save_path, exist_ok=True)
vllm_tensor_parallel = args.vllm_tensor_parallel

## Ultrachat generation
total_num_of_scripts = args.total_num_of_scripts
for i in range(total_num_of_scripts):
    output_dir = f"{data_save_folder_name}/ultrachat"
    json_save_path = f"{output_dir}/{i}_{total_num_of_scripts}.jsonl"
    script = f"""
python speculator/data_generation/vllm_data_generation.py --model={model_name} --tensor_parallel={vllm_tensor_parallel} --tokenizer={tokenizer_name} --cur_split={i} --output_dataset_path={json_save_path} --total_split={total_num_of_scripts}
    """
    with open(f"{script_save_path}/{data_save_folder_name}_{i:02}.sh", "w") as f:
        f.write(script)

## Magicoder generation
for i in range(total_num_of_scripts):
    output_dir = f"{data_save_folder_name}/magicoder"
    json_save_path = f"{output_dir}/{i}_{total_num_of_scripts}.jsonl"
    script = f"""
python speculator/data_generation/vllm_data_generation.py --hf_dataset magicoder --model={model_name} --tensor_parallel={vllm_tensor_parallel} --tokenizer={tokenizer_name} --cur_split={i} --output_dataset_path={json_save_path} --total_split={total_num_of_scripts}
    """
    with open(f"{script_save_path}/{data_save_folder_name}_magic_{i:02}.sh", "w") as f:
        f.write(script)
