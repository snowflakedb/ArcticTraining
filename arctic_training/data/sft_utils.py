import sys
from typing import Dict
from typing import List

import tqdm
from datasets import Dataset


def packing_sft_dataset(
    dataset, seed: int, rank: int, max_length: int, always_max_length: bool
):
    # packing for sft / cpt are different
    dataset = dataset.shuffle(seed=seed + rank)
    train_dataset: Dict[str, List] = {
        "input_ids": [],
        "labels": [],
        "position_ids": [],
        # "attention_mask": [],
    }
    example: Dict[str, List] = {
        "input_ids": [],
        "labels": [],
        "position_ids": [],
        # "attention_mask": [],
    }
    # pack multiple samples into one sample
    # for data in dataset:
    # TODO: make it multi-process?
    for data in tqdm.tqdm(
        dataset,
        total=len(dataset),
        dynamic_ncols=True,
        file=sys.stdout,
        desc="Packing data",
        disable=rank != 0,
    ):
        input_ids = data["input_ids"]
        # attention_mask = data["attention_mask"]
        labels = data["labels"]

        if (
            not always_max_length
            and len(example["input_ids"]) + len(input_ids) > max_length
        ) or len(example["input_ids"]) > max_length:
            train_dataset["input_ids"].append(example["input_ids"])
            train_dataset["labels"].append(example["labels"])
            train_dataset["position_ids"].append(example["position_ids"])
            # train_dataset["attention_mask"].append(example["attention_mask"])

            example = {
                "input_ids": [],
                "labels": [],
                "position_ids": [],
                # "attention_mask": [],
            }

        example["input_ids"].extend(input_ids)
        example["labels"].extend(labels)
        example["position_ids"].extend(list(range(len(input_ids))))
        # example["attention_mask"].extend(attention_mask)
    # add the last example
    if example["input_ids"]:
        train_dataset["input_ids"].append(example["input_ids"])
        train_dataset["labels"].append(example["labels"])
        train_dataset["position_ids"].append(example["position_ids"])
        # train_dataset["attention_mask"].append(example["attention_mask"])

    return Dataset.from_dict(train_dataset)
