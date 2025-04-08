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

from datasets import load_dataset
from vllm import SamplingParams

from arctic_training.synth import MultiReplicaVllmSynth


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process dataset configuration.")

    # Add command-line arguments
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="ultrachat",
        help="Path to your UltraChat dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
        help="Model name or path",
    )
    parser.add_argument(
        "--tensor_parallel",
        type=int,
        default=1,
        help="Number of tensor parallelism splits",
    )
    parser.add_argument("--output_path", type=str, default=None, help="Output path for Results")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens to generate")

    # Parse the arguments
    args = parser.parse_args()

    return args


def create_prompts(args, dataset):
    if args.hf_dataset == "ultrachat":
        prompts = [entry["prompt"] for entry in dataset if "prompt" in entry]
    elif args.hf_dataset == "magicoder":
        prompts = [entry["problem"] for entry in dataset if "problem" in entry]
    else:
        assert False, "In correct dataset argument."
    return prompts


# Load dataset (Hugging Face format)
def load_hf_dataset(dataset):
    if dataset == "ultrachat":
        return load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            num_proc=32,
        ).select_columns(["prompt", "messages"])
    elif dataset == "magicoder":
        return load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            num_proc=32,
        ).select_columns(["problem"])
    elif dataset == "magicoder-evol":
        return load_dataset(
            "ise-uiuc/Magicoder-Evol-Instruct-110K",
            split="train",
            num_proc=32,
        ).select_columns(["instruction"])

    else:
        print(f"Dataset {dataset} not supported")
        exit(0)


# Create a batch task from prompts and client
def create_batch_task(client, prompts, task_name):
    for prompt in prompts:
        client.add_chat_to_batch_task(
            task_name=task_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )


def main():
    # parse arguments from command line
    args = parse_arguments()

    client = MultiReplicaVllmSynth(
        model_params={"model": args.model},
        sampling_params=SamplingParams(temperature=0, max_tokens=args.max_tokens),
        tensor_parallel=args.tensor_parallel,
        work_dir=args.output_path,
    )

    # Load dataset
    dataset = load_hf_dataset(args.hf_dataset)

    # Extract prompts from the dataset
    prompts = create_prompts(args, dataset)

    # Create the task for the client using prompts
    task_name = f"task_{args.hf_dataset}"
    create_batch_task(client, prompts[:200], task_name)

    # execute the task and save the results
    client.execute_batch_task(task_name)

    # teardown all the replicas
    client.teardown()


if __name__ == "__main__":
    main()
