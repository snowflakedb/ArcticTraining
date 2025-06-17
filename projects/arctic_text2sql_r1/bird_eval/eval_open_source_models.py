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
import os


def main():
    parser = argparse.ArgumentParser(description="Run bird eval over one or more models")
    parser.add_argument(
        "--models", nargs="+", required=True, help="One or more model directories or HF IDs to evaluate"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input file containing prompts for evaluation"
    )
    parser.add_argument("--parallel_generation", action="store_true", help="If set, enables parallel generation mode")
    parser.add_argument(
        "--gold_file_path",
        type=str,
        default="/data/bohan/bird_submission/bird/dev_20240627/dev.json",
        help="Path to the gold file for evaluation",
    )
    parser.add_argument(
        "--dp_path",
        type=str,
        default="/data/bohan/bird_submission/bird/dev_20240627/dev_databases",
        help="Path to the database directory for evaluation",
    )
    parser.add_argument(
        "--self_consistency",
        action="store_true",
        help="If set, runs majority-voting (n=8, temp=0.8); otherwise greedy (n=1, temp=0.0)",
    )

    args = parser.parse_args()

    for model in args.models:
        # make a safe name for outputs
        model_name = model.replace("/", "_")
        eval_name = f"{model_name}_dev_bird"

        # pick devices
        if args.parallel_generation:
            visible_devices = "0,1,2,3,4,5,6,7"
            extra_flag = "--parallel_generation"
        else:
            extra_flag = ""
            if "7b" in model_name.lower():
                visible_devices = "0,1,2,3"
            else:
                visible_devices = "0,1,2,3,4,5,6,7"

        tensor_parallel_size = len(visible_devices.split(","))

        # set n & temperature
        if args.self_consistency:
            n = 8
            temp = 0.8
        else:
            n = 1
            temp = 0.0

        cmd = (
            f"python3 bird_eval/auto_evaluation.py {extra_flag} "
            f"--model_dir {model} "
            f"--input_file {args.input_file} "
            f"--eval_name {eval_name} "
            f"--visible_devices {visible_devices} "
            f"--tensor_parallel_size {tensor_parallel_size} "
            f"--n {n} "
            f"--temperature {temp} "
            f"--gold_file {args.gold_file_path} "
            f"--db_path {args.dp_path}"
        )

        print(f"\nRunning:\n  {cmd}\n")
        os.system(cmd)


if __name__ == "__main__":
    main()
