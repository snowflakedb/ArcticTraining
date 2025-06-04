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
import glob
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from functools import partial
from multiprocessing import Process
from multiprocessing import Queue


def rm_ext(filename):
    filename = os.path.splitext(filename)[0]
    return filename


def get_filename(file: str, remove_ext=True):
    if file.endswith("/"):
        file = file[:-1]
    name = os.path.split(file)[1]
    if remove_ext:
        return rm_ext(name)
    return name


get_filename = partial(get_filename, remove_ext=False)

wait_interval = 1


def start(sh, gpu, gpu_queue, end_queue):
    print(f"running {sh} on {gpu}", file=sys.stderr)
    env = os.environ
    env["CUDA_VISIBLE_DEVICES"] = gpu

    script = f"{os.environ['SHELL']} {sh}"
    subprocess.call(script.split(), env=env)
    print(f"done subprocess call {script}")
    gpu_queue.put(gpu)
    end_queue.put(sh)


def get_shs(folder):
    return sorted(glob.glob(os.path.join(folder, "*.sh")))


def run_multigpu(sh_folder, gpu_queue):
    end_queue = Queue()

    tmp_folder = sh_folder + "_tmp"
    os.makedirs(tmp_folder, exist_ok=True)

    procs = []
    while True:
        shs = get_shs(sh_folder)
        for sh in shs:
            dummy_path_to_indicate_its_run = os.path.join(tmp_folder, get_filename(sh))
            already_run = os.path.exists(dummy_path_to_indicate_its_run)
            if not already_run:
                gpu = gpu_queue.get()  # blocking wait
                proc = None
                try:
                    already_run = os.path.exists(dummy_path_to_indicate_its_run)
                    if already_run:
                        gpu_queue.put(gpu)
                        break
                    shutil.copy(sh, dummy_path_to_indicate_its_run)
                    sh_path = os.path.relpath(sh, os.getcwd())
                    proc = Process(target=start, args=(sh_path, gpu, gpu_queue, end_queue))
                    procs.append(proc)
                    proc.start()
                except Exception:
                    if proc is not None:
                        proc.join()
                    else:
                        gpu_queue.put(gpu)
                break  # if ran something, check another cfg!

        shs = get_shs(sh_folder)
        dummy_shs = get_shs(tmp_folder)
        if not args.inf and (
            end_queue.qsize() == len(procs) and set(map(get_filename, shs)) == set(map(get_filename, dummy_shs))
        ):
            break
        else:
            time.sleep(wait_interval)

    for proc in procs:
        proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sh_folder", type=str)
    parser.add_argument("--max_gpus", default=1, type=int)
    parser.add_argument("-s", "--start_gpu", default=0, type=int)
    parser.add_argument("--gpus", default=None, type=str)
    parser.add_argument("--inf", default=False, action="store_true", help="watch folder infinitely")
    parser.add_argument("-j", "--proc_num_for_each_gpu", default=1, type=int)
    parser.add_argument("-n", "--gpus_per_each_proc", default=1, type=int)
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn")
    gpu_queue = Queue()

    if args.gpus is not None:
        total_gpus = args.gpus.split(",")
    else:
        try:
            gpu_nums = args.max_gpus
        except Exception:
            print("nvidia-smi not found. Setting CUDA_VISIBLE_DEVICES as -1")
            args.start_gpu = -1
            gpu_nums = args.max_gpus = 1
        total_gpus = list(map(lambda x: str(x), list(range(args.start_gpu, args.start_gpu + gpu_nums))))

    assert len(total_gpus) % args.gpus_per_each_proc == 0
    n = args.gpus_per_each_proc
    chunked_gpus = [total_gpus[i : i + n] for i in range(0, len(total_gpus), n)]

    for _ in range(args.proc_num_for_each_gpu):
        for gpus_list in chunked_gpus:
            gpu_queue.put(",".join(gpus_list))

    sh_folder = args.sh_folder
    if sh_folder.endswith(os.sep):
        sh_folder = sh_folder[: -len(os.sep)]

    run_multigpu(sh_folder, gpu_queue)
