import glob
import json
import os

from datasets import Dataset
from tqdm.auto import tqdm

location = "llama3.1_gen_mlpspec_nodetok"
save_location = "llama3.1_gen_mlpspec_nodetok_disk"
location = "swiftkv_llama33_gen_mlpspec_nodetok_hf"
save_location = "swiftkv_llama33_gen_mlpspec_nodetok_disk"
total_data = {
    "input_ids": [],
    "labels": [],
}
for f in tqdm(list(sorted(glob.glob(os.path.join(location, "*/*/*.jsonl"))))):
    for line in open(f):
        data = json.loads(line)
        outputs = data.pop("output")
        assert len(outputs) == 256
        # inputs = data.pop('input')
        # inputs = inputs.view(-1, inputs.shape[-1])
        total_data["input_ids"].append(outputs)
        total_data["labels"].append(outputs)

print("Doing from_dict...")
dataset = Dataset.from_dict(total_data)
print("Done")
dataset.save_to_disk(save_location)
