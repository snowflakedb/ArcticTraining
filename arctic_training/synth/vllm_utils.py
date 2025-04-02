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
import signal
import subprocess
import time

import requests  # type:ignore

from arctic_training import logger


def check_vllm_servers_health(base_port, num_services):

    all_ids = set(range(num_services))
    ready_ids = set([])
    # Replace with your server's host and port
    while True:
        remaining_ids = all_ids - ready_ids
        logger.info(f"All Ids {all_ids}, Ready Ids {ready_ids}, Not Ready Ids {remaining_ids} ")
        if len(remaining_ids) == 0:
            break

        logger.info("Waiting for 30 seconds ...")
        time.sleep(30)
        for i in [id for id in remaining_ids]:
            url = f"http://localhost:{base_port + i}/health"
            try:
                logger.info(f"Checking Status of {url}")
                response = requests.get(url)
                if response.status_code == 200:
                    logger.info(f"VLLM server {url} is running.")
                    ready_ids.add(i)
                else:
                    logger.info(f"VLLM server {url} is not running. Status code: {response.status_code}")
            except requests.ConnectionError:
                logger.info(f"Failed to connect to the VLLM server {url}. It might not be running.")
            except Exception as e:
                logger.info(f"An error occurred: {e} at {url}")

    logger.info("All services are running")


"""Given model_name, tensor_parallelism, and gpu_ids,
this method will launch len(gpu_ids)/tensor_parallelism number of
vllm services each running with the given tensor_parallelism degree.
It will return the process_ids of the services, along with vllm_url
"""


def launch_vllm_servers(model_name, tensor_parallelism, gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], skip_launch=False):
    """
    Launches multiple VLLM services based on tensor parallelism and GPU IDs.

    Args:
        model_name (str): Name or path to the model to load.
        tensor_parallelism (int): Number of GPUs per service.
        gpu_ids (list[int]): List of available GPU IDs.

    Returns:
        dict: A dictionary with service process IDs and URLs, e.g.:
            {
                "process_ids": [pid1, pid2, ...],
                "vllm_urls": ["http://localhost:8000", "http://localhost:8001", ...]
            }
    """
    num_gpus = len(gpu_ids)
    if num_gpus % tensor_parallelism != 0:
        raise ValueError("Number of GPUs must be divisible by tensor_parallelism.")

    # Number of services to launch
    num_services = num_gpus // tensor_parallelism

    # Track processes and URLs
    processes = []
    urls = []

    # Port number to start from
    base_port = 8000

    # Just return the urls, the assumption is that launch as happened already
    if skip_launch:
        urls = [f"http://localhost:{base_port + i}/v1/chat/completions" for i in range(num_services)]
        return processes, urls

    # Loading the model in CPU. This will download the model if it has not been already.
    # Avoids multiple process from downloading the same model below
    logger.info("Downloading Model")
    from transformers import AutoModelForCausalLM

    _ = AutoModelForCausalLM.from_pretrained(model_name)
    logger.info("Done Downloading Model")
    for i in range(num_services):
        # GPUs assigned to this service
        assigned_gpus = gpu_ids[i * tensor_parallelism : (i + 1) * tensor_parallelism]

        # Set CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpus))

        # Define the command for starting VLLM
        command = [
            "vllm",
            "serve",
            model_name,
            "--tensor-parallel-size",
            str(tensor_parallelism),
            "--port",
            str(base_port + i),
            "--swap_space",
            str(16),
            "--enable-chunked-prefill",
            "--use-v2-block-manager",
            "--disable-log-requests",
        ]

        # Start the VLLM service
        process = subprocess.Popen(command)
        processes.append(process.pid)
        urls.append(f"http://localhost:{base_port + i}/v1/chat/completions")

    check_vllm_servers_health(base_port, num_services)
    logger.info(f"Created Processs: {processes}")
    logger.info(f"Created VLLM Servers: {urls}")

    return processes, urls


def kill_processes(process_ids):
    for pid in process_ids:
        try:
            # Send SIGTERM signal to the process
            os.kill(pid, signal.SIGTERM)
            logger.info(f"Process {pid} terminated gracefully.")
        except ProcessLookupError:
            logger.info(f"Process {pid} does not exist.")
        except PermissionError:
            logger.info(f"Permission denied to terminate process {pid}.")
        except Exception as e:
            logger.info(f"Error terminating process {pid}: {e}")
