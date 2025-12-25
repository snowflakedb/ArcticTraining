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
"""
TODO: update bash file to replace /data_generation

Examples:
  python arctic_forge_data_generation.py \
    --datasets magicoder ultrachat \
    --model_path /data-fast/model \
    --output_dir /data-fast/data_xxx \
    --concat_output_dir /data-fast/data_xxx_hf_disk
"""

import argparse
import atexit
import gc
import os
from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import ray
import torch
from arctic_forge import Driver
from arctic_forge.config import ModelConfig
from datasets import Dataset
from datasets import concatenate_datasets
from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def load_template(template_name: str) -> str:
    """Load a template file from the same directory as this module."""
    current_dir = Path(__file__).parent
    template_path = current_dir / template_name
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_ultrachat() -> Dataset:
    return load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
        num_proc=32,
    )


def _load_magicoder() -> Dataset:
    result = load_dataset(
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        split="train",
        num_proc=32,
    )

    def instruct_format_conversation(
        example: Dict[str, Any],
        query_key: str,
        response_key: str,
        source_name: str,
    ) -> Dict[str, Any]:
        conversation = [
            {"role": "user", "content": example[query_key]},
            {"role": "assistant", "content": example[response_key]},
        ]
        return {"source": source_name, "messages": conversation}

    return result.map(
        partial(
            instruct_format_conversation,
            query_key="problem",
            response_key="solution",
            source_name="Magicoder",
        )
    )


def load_hf_dataset(dataset: str, args: argparse.Namespace) -> Dataset:
    """Replicates the old vllm_data_generation.load_hf_dataset logic."""
    if dataset == "ultrachat":
        return _load_ultrachat()
    if dataset == "magicoder":
        return _load_magicoder()

    raise ValueError(f"Dataset {dataset} not supported")


def build_prompt_segments(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    gen_prompt_length: int,
    num_proc: int = 16,
) -> Dataset:
    """
    Turn conversation-style samples into fixed-length prompt segments using multiprocessing.
    """

    def process_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        chat_strs = []
        for messages in batch["messages"]:
            try:
                s = tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                s = tokenizer.apply_chat_template(
                    conversation=messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            chat_strs.append(s)

        tokenized_batch = tokenizer(
            chat_strs,
            add_special_tokens=False,
        )

        new_input_tokens = []
        new_prompt_texts = []

        for input_ids in tokenized_batch["input_ids"]:
            if not input_ids:
                continue

            max_len = (len(input_ids) // gen_prompt_length) * gen_prompt_length
            if max_len == 0:
                continue

            for start in range(0, max_len, gen_prompt_length):
                chunk_ids = input_ids[start : start + gen_prompt_length]
                new_input_tokens.append(chunk_ids)
                new_prompt_texts.append(tokenizer.decode(chunk_ids, skip_special_tokens=False))

        return {"input_tokens": new_input_tokens, "prompt_text": new_prompt_texts}

    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Building segments (Multi-threaded)",
    )

    if len(processed_dataset) == 0:
        raise RuntimeError("No prompt segments constructed. Check inputs.")

    return processed_dataset


def concatenate_and_save_dataset(
    data_save_folder_name: str, disk_save_location: str, tokenizer: AutoTokenizer = None
) -> None:
    """
    Loads datasets, concatenates them, and robustly unpacks ArcticForge results
    to map columns for Speculator training.
    """
    print(f"[INFO] Concatenating datasets from {data_save_folder_name}...")

    subdirs = [f.path for f in os.scandir(data_save_folder_name) if f.is_dir()]
    datasets_to_concat = []

    for subdir in tqdm(subdirs, desc="Loading datasets"):
        try:
            ds = load_from_disk(subdir)
            datasets_to_concat.append(ds)
        except Exception as e:
            print(f"[WARN] Could not load dataset from {subdir}: {e}")

    if not datasets_to_concat:
        print("[WARN] No valid datasets found. Skipping concatenation.")
        return

    print(f"[INFO] Concatenating {len(datasets_to_concat)} datasets...")
    merged_dataset = concatenate_datasets(datasets_to_concat)

    available_columns = merged_dataset.column_names
    print(f"[DEBUG] Merged dataset columns: {available_columns}")

    def format_for_speculator(example):
        output_ids = None

        if "__arctic_forge_result__" in example:
            result = example["__arctic_forge_result__"]
            if isinstance(result, dict):
                if "input_ids" in result:
                    output_ids = result["input_ids"]
                elif "output" in result:
                    output_ids = result["output"]
                elif "response_text" in result and tokenizer:
                    output_ids = tokenizer(result["response_text"], add_special_tokens=False)["input_ids"]

        if output_ids is None:
            if "output" in example and example["output"]:
                output_ids = example["output"]
            elif "response_text" in example and tokenizer:
                output_ids = tokenizer(example["response_text"], add_special_tokens=False)["input_ids"]
            elif "response" in example and tokenizer:
                output_ids = tokenizer(example["response"], add_special_tokens=False)["input_ids"]

        if output_ids is None:
            # Print keys to help debug if this happens again
            raise ValueError(f"Could not extract output IDs. Row keys: {list(example.keys())}")

        return {"input_ids": output_ids, "labels": output_ids}

    print("[INFO] Formatting columns: unpacking and mapping to 'input_ids'/'labels'...")

    merged_dataset = merged_dataset.map(format_for_speculator, num_proc=32, remove_columns=available_columns)

    print(f"[INFO] Saving merged dataset to: {disk_save_location}")
    merged_dataset.save_to_disk(disk_save_location)
    print("[INFO] Concatenation complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end data generation pipeline using ArcticForge. Can run one or many HF datasets concurrently."
        ),
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="ultrachat",
        help="Single dataset name if --datasets is not provided.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="If provided, run these HF datasets concurrently; overrides --hf_dataset. Examples: ultrachat magicoder",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Base model name or path passed to ArcticForge Driver.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory. In multi-dataset mode, each dataset will write to <output_dir>/<dataset_name>.",
    )
    parser.add_argument(
        "--concat_output_dir",
        type=str,
        default=None,
        help=(
            "If provided, all JSONL files in `output_dir` (including subdirectories) "
            "will be concatenated into a single HF Dataset and saved here."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Checkpoint folder (for single dataset) or checkpoint ROOT (for multi-dataset). "
            "In multi-dataset mode, checkpoints go to <checkpoint_path>/<dataset_name>."
        ),
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=10000,
        help="Checkpointing frequency (in samples) for ArcticForge.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max tokens to generate from the base model.",
    )
    parser.add_argument(
        "--gen_prompt_length",
        type=int,
        default=64,
        help="Prompt context length in tokens (matches old pipeline).",
    )
    parser.add_argument(
        "--gen_temp",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--gen_top_p",
        type=float,
        default=0.8,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--gen_top_k",
        type=int,
        default=20,
        help="Top-k sampling parameter.",
    )
    return parser.parse_args()


def start_task_for_dataset(
    driver: Driver,
    dataset_name: str,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> str:
    """Load HF dataset, build segments, and submit run to the existing Driver."""

    print(f"[INFO] Loading HF dataset: {dataset_name}")
    base_dataset = load_hf_dataset(dataset_name, args)

    print(f"[INFO] Building prompt segments for dataset: {dataset_name}")
    prompt_dataset = build_prompt_segments(
        dataset=base_dataset,
        tokenizer=tokenizer,
        gen_prompt_length=args.gen_prompt_length,
    )

    # Dataset-specific output directory
    if args.datasets:
        out_dir = os.path.join(args.output_dir, dataset_name)
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Dataset-specific checkpoint path
    run_kwargs: Dict[str, Any] = {}
    if args.checkpoint_path is not None:
        if args.datasets:
            ckpt_dir = os.path.join(args.checkpoint_path, dataset_name)
        else:
            ckpt_dir = args.checkpoint_path
        os.makedirs(ckpt_dir, exist_ok=True)
        run_kwargs["checkpoint_path"] = ckpt_dir
        run_kwargs["checkpoint_frequency"] = args.checkpoint_frequency

    # Define the async processing pipeline for this dataset
    # We define this dynamically on the shared driver instance
    @driver.pipeline
    async def generation_pipeline(sample: Dict[str, Any], llm) -> Dict[str, Any]:
        import sys
        import traceback

        try:
            request_output = await llm.generate(
                prompt=sample["prompt_text"],
                max_tokens=args.max_tokens,
                temperature=args.gen_temp,
                top_p=args.gen_top_p,
                top_k=args.gen_top_k,
                ignore_eos=True,
            )

            # Note: OpenAI API does not return token id directly
            response_text = str(request_output)
            output_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

            target_len = args.max_tokens
            current_len = len(output_ids)

            if current_len > target_len:
                # Case A: Too long (e.g. 320 tokens or 257). Keep the LAST 256 tokens.
                output_ids = output_ids[-target_len:]
            elif current_len < target_len:
                # Case B: Too short (e.g. 255). Pad with zeros to exactly 256.
                padding = [0] * (target_len - current_len)
                output_ids = list(output_ids) + padding

            return {
                "input_ids": output_ids,
                "labels": output_ids,
            }
        except Exception:
            traceback.print_exc(file=sys.stderr)
            raise

    print(f"[INFO] Starting ArcticForge run for dataset: {dataset_name}")
    # This will use the already initialized Ray/vLLM instance
    driver.run(prompt_dataset, **run_kwargs)

    return out_dir


def main() -> None:
    args = parse_args()

    tokenizer_name = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Determine which datasets to run
    if args.datasets:
        dataset_names = args.datasets
        print(f"[INFO] Queuing multiple datasets: {dataset_names}")
    else:
        dataset_names = [args.hf_dataset]
        print(f"[INFO] Queuing single dataset: {dataset_names[0]}")

    model_config = ModelConfig(
        model_name_or_path=args.model_path,
        tensor_parallel_size=1,
        max_model_len=8192,
        max_num_seqs=2048,
        async_scheduling=True,
    )

    # Process datasets loop
    for ds_name in dataset_names:
        print(f"\n{'='*40}")
        print(f"[INFO] Starting cycle for dataset: {ds_name}")
        print(f"{'='*40}")

        # 1. Initialize Driver (Starts Ray + vLLM)
        print(f"[INFO] Initializing ArcticForge Driver for model: {args.model_path}")
        driver = Driver(config=model_config)
        atexit.unregister(driver.shutdown)

        try:
            # 2. Run the task
            out_dir = start_task_for_dataset(driver, ds_name, tokenizer, args)

            # 3. Save results
            print(f"[INFO] Saving results for dataset: {ds_name}")
            driver.save_results(out_dir)
            print(f"[INFO] Results saved to: {out_dir}")

        except Exception as e:
            print(f"[ERROR] Failed processing dataset {ds_name}: {e}")
            raise e  # Re-raise after cleanup if you want the whole script to stop

        finally:
            # 4. ELEGANT SHUTDOWN & RESOURCE CLEANUP
            print(f"[INFO] Shutting down resources for {ds_name}...")

            # Check if the driver has a built-in shutdown method
            if hasattr(driver, "shutdown"):
                driver.shutdown()

            # Delete the python object
            del driver

            # Force garbage collection to free python memory
            gc.collect()

            # Shut down Ray to kill vLLM actors (Frees GPU Memory)
            if ray.is_initialized():
                print("[INFO] Shutting down Ray instance...")
                ray.shutdown()

            # Clear any residual local CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[INFO] Cleanup complete for {ds_name}.\n")

    # 5. Concatenate results if requested
    if args.concat_output_dir:
        print("\n[INFO] Starting dataset concatenation...")
        concatenate_and_save_dataset(data_save_folder_name=args.output_dir, disk_save_location=args.concat_output_dir)


if __name__ == "__main__":
    main()
