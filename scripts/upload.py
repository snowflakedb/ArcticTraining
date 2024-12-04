import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import time
import glob
import requests
import argparse

from huggingface_hub import HfApi
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model-name", 
                    help="model name: snowflake/snowflake-arctic-instruct or snowflake/snowflake-arctic-base", 
                    type=str, required=True)
parser.add_argument("-s", "--src-path", help="local path of model ckpts", required=True)
args = parser.parse_args()

hf_token = os.environ.get("HF_TOKEN")

api = HfApi()

# keep track of successfully upload files, helpful in case the upload dies
if not os.path.isfile('success'):
    with open('success', 'w'):
        pass

success_files = []
with open('success', 'r') as fd:
    for line in fd.readlines():
        success_files.append(line.strip())
print(f"previously uploaded files: {success_files}")

def ts():
    current_timestamp = time.time()
    current_datetime = datetime.fromtimestamp(current_timestamp)
    return current_datetime.strftime("%Y-%m-%d %H:%M:%S")

def success(local_path):
    print(f'{ts()} successfully uploaded {local_path}')
    with open('success', 'a') as fd:
        fd.write(f"{local_path}\n")
    success_files.append(local_path)

def upload(local_path):
    name = os.path.basename(local_path)
    print(f'{ts()} uploading {name}...')
    for i in range(10):
        try:
            results = api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=name,
                repo_id=args.model_name,
                repo_type="model",
                token=hf_token
            )
            success(local_path)
            return
        except requests.HTTPError as err:
            print(err)
            print(f"{ts()} {local_path} failed upload, attempt {i}")
            time.sleep(60)
        except RuntimeError as err:
            print(err)
            print(f"{ts()} {local_path} failed upload with RuntimeError, attempt {i}")
            time.sleep(60)

# meta data and tokenizer files
metadata_and_tokenizer_files = [
    'config.json', 
    'generation_config.json', 
    'model.safetensors.index.json', 
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer_config.json'
]
metadata_and_tokenizer_files = [os.path.join(args.src_path, p) for p in metadata_and_tokenizer_files]

# core checkpoints
ckpt_files = glob.glob(os.path.join(args.src_path, '*safetensors'))
ckpt_files.sort()

upload_files = metadata_and_tokenizer_files + ckpt_files

# ensure all expected files exist
for p in upload_files:
    assert os.path.isfile(p), f"expected file doesn't exist {p}"

# upload everything
for local_path in upload_files:
    if local_path in success_files:
        print(f'skipping {local_path}')
    else:
        upload(local_path)