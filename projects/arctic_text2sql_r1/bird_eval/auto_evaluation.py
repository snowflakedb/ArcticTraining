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
import time

import evaluate_bird
import matplotlib.pyplot as plt
from tqdm import tqdm


def visualize(eval_name, acc_dict, mode, out_dir):
    """
    mode: 'greedy_search' or 'major_voting'
    """
    plt.figure(figsize=(10, 6))
    x = list(range(len(acc_dict)))
    y = list(acc_dict.values())
    plt.plot(x, y, marker="o", linestyle="-", label=mode.replace("_", " ").title())
    plt.title(f"{eval_name} — {mode.replace('_', ' ').title()} Accuracy")
    plt.xlabel("checkpoint index")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{mode}.png"))
    plt.close()


def save_json(d, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, default="./ckpts", help="Either local folder of checkpoints or a HF model ID"
    )
    parser.add_argument(
        "--multiple_models", action="store_true", help="If set, look for subfolders in `model_dir` named `ckpt-<idx>`"
    )
    parser.add_argument(
        "--eval_name", type=str, required=True, help="A short name for this evaluation (used in output paths)"
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to JSON input prompts")
    parser.add_argument("--gold_file", type=str, required=True, help="Path to gold SQL file")
    parser.add_argument("--db_path", type=str, required=True, help="Path to directory of SQLite DBs")
    parser.add_argument("--visible_devices", type=str, default="0,1", help="CUDA_VISIBLE_DEVICES for generation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--n", type=int, default=1, help="Number of samples per prompt (1=greedy, >1=major_voting)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--parallel_generation", action="store_true", help="Pass --parallel_generation to infer.py")
    opt = parser.parse_args()

    # Build list of checkpoint IDs
    if opt.multiple_models:
        # Expect dirs like ckpt-0, ckpt-1, etc.
        ckpts = sorted(os.listdir(opt.model_dir), key=lambda x: int(x.split("-")[-1]))
    else:
        ckpts = [""]  # single run, empty string

    # Prepare output dirs
    results_dir = os.path.join("results", opt.eval_name)
    eval_dir = os.path.join("evaluation_results", opt.eval_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    extra_flag = "--parallel_generation" if opt.parallel_generation else ""

    greedy_acc = {}
    major_acc = {}

    for idx, ckpt_id in enumerate(tqdm(ckpts, desc="Checkpoints")):
        # 1) Resolve model path/ID
        if ckpt_id:
            model_path = os.path.join(opt.model_dir, ckpt_id)
        else:
            model_path = opt.model_dir

        ### Greedy Search ###
        gs_json = os.path.join(results_dir, f"greedy_search_{ckpt_id or 'base'}.json")
        if not os.path.exists(gs_json):
            gs_temp = 0.0 if opt.n > 1 else opt.temperature
            cmd = (
                f"CUDA_VISIBLE_DEVICES={opt.visible_devices} "
                f"python3 bird_eval/infer.py {extra_flag} "
                f"--pretrained_model_name_or_path {model_path} "
                f"--input_file {opt.input_file} "
                f"--output_file {gs_json} "
                f"--tensor_parallel_size {opt.tensor_parallel_size} "
                "--n 1 "
                f"--temperature {gs_temp}"
            )
            start = time.time()
            os.system(cmd)
            print(f"[{ckpt_id or 'base'}] Greedy gen took {time.time()-start:.1f}s")
        else:
            print(f"[{ckpt_id or 'base'}] Skipping greedy (exists)")

        # Evaluate greedy
        gs_acc, _ = evaluate_bird.run_eval(
            opt.gold_file, gs_json, opt.db_path, mode="greedy_search", save_pred_sqls=True
        )
        greedy_acc[ckpt_id] = gs_acc

        # Save & plot
        save_json(greedy_acc, os.path.join(eval_dir, "greedy_search.json"))
        visualize(opt.eval_name, greedy_acc, "greedy_search", eval_dir)

        ### Major Voting (only if n>1) ###
        if opt.n > 1:
            mv_json = os.path.join(results_dir, f"major_voting_{ckpt_id or 'base'}.json")
            if not os.path.exists(mv_json):
                cmd = (
                    f"CUDA_VISIBLE_DEVICES={opt.visible_devices} "
                    f"python3 bird_eval/infer.py {extra_flag} "
                    f"--pretrained_model_name_or_path {model_path} "
                    f"--input_file {opt.input_file} "
                    f"--output_file {mv_json} "
                    f"--tensor_parallel_size {opt.tensor_parallel_size} "
                    f"--n {opt.n} "
                    f"--temperature {opt.temperature}"
                )
                start = time.time()
                os.system(cmd)
                print(f"[{ckpt_id or 'base'}] Major Voting gen took {time.time()-start:.1f}s")
            else:
                print(f"[{ckpt_id or 'base'}] Skipping major voting (exists)")

            mv_acc, _ = evaluate_bird.run_eval(
                opt.gold_file, mv_json, opt.db_path, mode="major_voting", save_pred_sqls=True
            )
            major_acc[ckpt_id] = mv_acc

            # Save & plot
            save_json(major_acc, os.path.join(eval_dir, "major_voting.json"))
            visualize(opt.eval_name, major_acc, "major_voting", eval_dir)

    print("✅ Done.")


if __name__ == "__main__":
    main()
