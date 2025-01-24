import json
import asyncio
import aiohttp
from datasets import load_dataset, Dataset
import os, subprocess, socket
import signal
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import time


import argparse

'''
1. Launch VLLM Services on all nodes with appropriate number of replicas based on the TP
2. Wait for VLLM Services to come live.
3. Run one generation process per node.
4. Within this process, create input prompts based on node ID, and round robin them across replicas within local node
5. Using the responses, create the output dataset and store them based on node ID.
6. Wait for all nodes to complete generation and dataset creation.
7. Combine all datasets into a single dataset.
'''

# # Configuration
# ULTRACHAT_DATASET_PATH = "HuggingFaceH4/ultrachat_200k"  # Replace with the path to your UltraChat dataset
# MODEL = 'neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8'
# TENSOR_PARALLEL = 2
# OUTPUT_DATASET_PATH = f"/checkpoint/users/samyam/datasets/synth/{MODEL}/ultrachat"  # Replace with the desired output path for the Hugging Face dataset
# MODEL = 'neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8'
# VLLM_SERVER_URL = "http://127.0.0.1:8000/v1/completions"  # Replace with the VLLM server URL
# BATCH_SIZE = 20  # Number of prompts to process in each batch
# HOSTFILE = 'hostfile.txt'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process UltraChat dataset configuration.")

    # Add command-line arguments
    parser.add_argument("--hf_dataset", type=str, default='ultra_chat', 
                        help="Path to your UltraChat dataset")
    parser.add_argument("--model", type=str, default="neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8", 
                        help="Model name or path")
    parser.add_argument("--tensor_parallel", type=int, default=2, 
                        help="Number of tensor parallelism splits")
    parser.add_argument("--output_dataset_path", type=str, 
                        default=None, 
                        help="Output path for the Hugging Face dataset")
    parser.add_argument("--batch_size", type=int, default=20, 
                        help="Number of prompts to process in each batch per node")
    parser.add_argument("--hostfile", type=str, default=None, 
                        help="Path to the hostfile")
    parser.add_argument("--skip_launch", 
                        action="store_true",  # This makes it a boolean flag
                        help="Skip the launch process if this flag is set. Handy if you just want to run generation and server have been launched already")
    parser.add_argument("--skip_generation", 
                    action="store_true",  # This makes it a boolean flag
                    help="Skip the generation if this flag is set. Handy if you just want to launch the vllm servers.")

    # Parse the arguments
    args = parser.parse_args()
    
    return args

import requests
def check_health(base_port,num_services):

    all_ids = set(range(num_services))
    ready_ids = set([])
    # Replace with your server's host and port
    while True:
        remaining_ids = all_ids - ready_ids
        print(f"All Ids {all_ids}, Ready Ids {ready_ids}, Not Ready Ids {remaining_ids} ")
        if len(remaining_ids) == 0:
            break

        print(f"Waiting for 30 seconds ...")
        time.sleep(30)
        for i in [id for id in remaining_ids]:    
            url = f"http://localhost:{base_port + i}/health"
            try:
                print(f"Checking Status of {url}")
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"VLLM server {url} is running.")
                    ready_ids.add(i)
                else:
                    print(f"VLLM server {url} is not running. Status code: {response.status_code}")
            except requests.ConnectionError:
                print(f"Failed to connect to the VLLM server {url}. It might not be running.")
            except Exception as e:
                print(f"An error occurred: {e} at {url}")
                
    print(f"All services are running")

'''Given model_name, tensor_parallelism, and gpu_ids, 
this method will launch len(gpu_ids)/tensor_parallelism number of
vllm services each running with the given tensor_parallelism degree.
It will return the process_ids of the services, along with vllm_url
''' 
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
    
    #Just return the urls, the assumption is that launch as happened already
    if skip_launch:
        urls = [f"http://localhost:{base_port + i}/v1/completions" for i in range(num_services)]
        return processes, urls
        
    # Loading the model in CPU. This will download the model if it has not been already.
    # Avoids multiple process from downloading the same model below
    print(f"Downloading Model")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"Done Downloading Model")
    for i in range(num_services):
        # GPUs assigned to this service
        assigned_gpus = gpu_ids[i * tensor_parallelism:(i + 1) * tensor_parallelism]
        
        # Set CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, assigned_gpus))
        
        # Define the command for starting VLLM
        command = [
            "vllm",
            "serve",
            model_name,
            "--tensor-parallel-size", str(tensor_parallelism),
            "--port", str(base_port + i),
            "--swap_space", str(16), 
            "--enable-chunked-prefill",
            "--use-v2-block-manager",
            "--disable-log-requests"
        ]
        
        # Start the VLLM service
        process = subprocess.Popen(command)
        processes.append(process.pid)
        urls.append(f"http://localhost:{base_port + i}/v1/completions")
        
    check_health(base_port, num_services)
    print(f"Created Processs: {processes}")
    print(f"Created VLLM Servers: {urls}")

    return processes, urls


def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        return load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="test_sft" if eval else "train_sft",
            num_proc=num_proc,
        ).select_columns(["messages"])


# Load dataset (Hugging Face format)
def load_hf_dataset(dataset):
    if dataset == 'ultrachat':
        return load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            num_proc=32,
        ).select_columns(["prompt"])
    elif dataset == 'magicoder':
        return load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            num_proc=32,
        ).select_columns(["problem"])
    else:
        print(f"Dataset {dataset} not supported")
        exit(0)
        
        
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
            
# Send a prompt to the VLLM server and get the response asynchronously
async def generate_response(args, session, prompt, vllm_url):
    payload = {
        "model": args.model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    
    async with session.post(vllm_url, json=payload) as response:
        if response.status == 200:
            generated_text=""
            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue

                chunk = remove_prefix(chunk_bytes.decode("utf-8"),
                                        "data: ")
                if not chunk == "[DONE]":
                    data = json.loads(chunk)
                    if data["choices"][0]["text"]:                       
                        generated_text += data["choices"][0]["text"]
                        
            result = generated_text                    
            return result
        else:
            print(f"Error: {response.status} - {await response.text()}")
            return ""

# Process prompts asynchronously across all the vllm replicas
def process_prompts(args, prompts, vllm_urls):
    num_urls = len(vllm_urls)
    async def process():
        timeout = aiohttp.ClientTimeout(total=1000) 
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [generate_response(args, session, prompt, vllm_urls[i % num_urls]) for i, prompt in enumerate(prompts)]
            return await asyncio.gather(*tasks)

    return asyncio.run(process())

# Save responses as a Hugging Face dataset
def save_as_huggingface_dataset(prompts, responses, output_path):
    assert output_path is not None, "Please provide an output_path"
    data = [{"prompt": prompt, "response": response} for prompt, response in zip(prompts, responses)]
    dataset = Dataset.from_dict({"prompt": [d["prompt"] for d in data], "response": [d["response"] for d in data]})
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")
    


def get_node_info(hostfile):
    """
    Get the number of hosts in the hostfile and the line number of the current host.

    Args:
        hostfile (str): Path to the hostfile.

    Returns:
        tuple: A tuple containing the total number of hosts (int) and the line number (int, 0-based index) of the current host. 
               If the current host is not found, the line number will be -1.
    """
    #Running on a single node with node ID 0
    if hostfile is None:
        return 1, 0
    
    try:
        # Get the current host's IP address
        current_host_ip = socket.gethostbyname(socket.gethostname())

        with open(hostfile, 'r') as file:
            lines = file.readlines()

        # Remove whitespace and empty lines
        hosts = [line.strip().split()[0] for line in lines if line.strip()]
        
        # Find the current host's line number (1-based index)
        try:
            line_number = hosts.index(current_host_ip)
        except ValueError:
            line_number = -1  # Host not found

        # Return the total number of hosts and the line number of the current host
        return len(hosts), line_number

    except FileNotFoundError:
        raise FileNotFoundError(f"The hostfile '{hostfile}' does not exist.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the hostfile: {e}")

def split_dataset(dataset, n, i):
    """
    Split a dataset as evenly as possible into n smaller datasets and return the dataset at index i.

    Args:
        dataset (list): The dataset to split.
        n (int): The number of smaller datasets to create.
        i (int): The index of the smaller dataset to return (0-based index).

    Returns:
        list: The i-th smaller dataset.
    """
    if n <= 0:
        raise ValueError("The number of splits 'n' must be greater than 0.")
    if i < 0 or i >= n:
        raise ValueError("The index 'i' must be between 0 and n-1.")

    # Calculate the size of each split
    length = len(dataset)
    split_size = length // n
    
    # Calculate the start and end indices for the i-th split
    start = i * split_size
    end = min(start + split_size, length)
    
    return start, end

def kill_processes(process_ids):
    for pid in process_ids:
        try:
            # Send SIGTERM signal to the process
            os.kill(pid, signal.SIGTERM)
            print(f"Process {pid} terminated gracefully.")
        except ProcessLookupError:
            print(f"Process {pid} does not exist.")
        except PermissionError:
            print(f"Permission denied to terminate process {pid}.")
        except Exception as e:
            print(f"Error terminating process {pid}: {e}")

def create_prompt(args, dataset):
    if args.hf_dataset == 'ultrachat':
        prompts = [entry["prompt"] for entry in dataset if "prompt" in entry]
    elif args.hf_dataset == 'magicoder':
        prompts = [entry["problem"] for entry in dataset if "problem" in entry]
    else:
        assert False, "In correct dataset argument."
    return prompts
        
        
def generate(args, process_ids, vllm_urls):
    
    # Number of nodes used in generation, and node ID
    node_size, node_id = get_node_info(args.hostfile)

    # Load dataset
    dataset = load_hf_dataset(args.hf_dataset)
        
    # Extract prompts from the dataset
    prompts = create_prompt(args, dataset)

    # Process in batches
    total_prompts = len(prompts)
    split_size = total_prompts // node_size
    start = node_id * split_size
    end = start + split_size
    
    all_responses = []

    for i in tqdm(range(start, end, args.batch_size), desc="Batch"):
        output_path = f"{args.output_dataset_path}/{node_id}/{i}_{split_size}"
        if not os.path.exists(output_path):
            batch_prompts = prompts[i:min(i + args.batch_size, total_prompts)]
            print(f"Processing Samples {i} ... batch {i // args.batch_size + 1} ({len(batch_prompts)} prompts)...", flush=True)
            batch_responses = process_prompts(args, batch_prompts, vllm_urls)
            all_responses.extend(batch_responses)
            
            # Save responses as a Hugging Face dataset storing each batch
            # Saving in batches allows for checkpointing
            save_as_huggingface_dataset(prompts, all_responses, output_path)
        
            
    if len(process_ids) > 0:
        kill_processes(process_ids)
        
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets

def combine_datasets(output_path):
    
    # Collect paths to all .arrow files
    arrow_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(output_path)
        for file in files if file.endswith('.arrow')
    ]

    # Load datasets from .arrow files and combine them
    datasets = [Dataset.from_file(file) for file in arrow_files]
    combined_dataset = concatenate_datasets(datasets)

    # Save the combined dataset to disk
    combined_dataset.save_to_disk(f"{output_path}/combined")
    print(f"Combined dataset saved to {output_path}/combined")
    
# Main function to process the dataset and generate responses in batches
def main():
    
    #parse arguments from command line
    args = parse_arguments()
    
    # Use combine_datasets to combine all the datasets into a single one
    if args.combine_datasets:
        combine_datasets(args)
        exit(0)
    
    #launch vllm servers or get the urls from a previous launch
    process_ids, vllm_urls = launch_vllm_servers(args.model, args.tensor_parallel, skip_launch=args.skip_launch)
    
    #Start the generation
    if not args.skip_generation:
        generate(args, process_ids, vllm_urls)
    
if __name__ == "__main__":
    main()
