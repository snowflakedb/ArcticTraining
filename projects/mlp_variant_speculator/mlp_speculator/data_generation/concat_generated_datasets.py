import glob
import json
import os

from datasets import Dataset
from tqdm.auto import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--data_save_folder_name", required=True)
parser.add_argument("--data_concat_folder_name", required=True)
args = parser.parse_args()

data_save_folder_name = args.data_save_folder_name
disk_save_location = args.data_concat_folder_name


total_data = {
    "input_ids": [],
    "labels": [],
}
all_jsonl_files = list(sorted(glob.glob(os.path.join(data_save_folder_name, "**/*.jsonl"), recursive=True)))
for f in tqdm(all_jsonl_files):
    for line in open(f):
        data = json.loads(line)
        outputs = data.pop("output")
        assert len(outputs) == 256
        total_data["input_ids"].append(outputs)
        total_data["labels"].append(outputs)

dataset = Dataset.from_dict(total_data)
dataset.save_to_disk(disk_save_location)
