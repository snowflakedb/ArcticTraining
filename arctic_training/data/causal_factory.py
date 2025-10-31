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

import itertools
from typing import Dict
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerBase

from arctic_training.config.data import DataConfig
from arctic_training.data.factory import DataFactory
from arctic_training.data.utils import DatasetType


class DataCollatorForCausalLM:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:

        input_ids = torch.stack(list(torch.tensor(example["input_ids"]) for example in instances))
        position_ids = torch.stack(list(torch.tensor(example["position_ids"]) for example in instances))
        packed_sample_seqlens = [example["packed_sample_seqlens"] for example in instances]

        labels = input_ids.clone()

        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "packed_sample_seqlens": packed_sample_seqlens,
        }

        return sample


def slice_and_pack_causal_batch(
    batch: Dict[str, List[List[int]]],
    max_length: int,
    overflow_buffer: List[List[int]],
) -> Dict[str, List[List[int]]]:
    """
    The overflow buffer is used to carry on any short samples through `map` calls. When all of dataset was processed the overflow buffer may contain 1 short sample and will be dropped (but can be fixed to pad and use it as well)
    """

    keys = ("input_ids", "position_ids", "packed_sample_seqlens")
    packed_batch: Dict[str, List[List[int]]] = {k: [] for k in keys}

    def build_sample(input_ids_group):
        """
        pass a list of input_ids
        """
        return dict(
            input_ids=list(itertools.chain(*input_ids_group)),
            position_ids=list(itertools.chain(*[range(len(ll)) for ll in input_ids_group])),
            packed_sample_seqlens=[len(ll) for ll in input_ids_group],
        )

    def add_sample_to_batch(sample):
        for k in keys:
            packed_batch[k].append(sample[k])

    def add_to_overflow_buffer(input_ids):
        overflow_buffer.append(input_ids)

    for input_ids in batch["input_ids"]:
        if len(input_ids) >= max_length:
            # 1. slice the incoming full text sample into max_length shards
            # 2. add shards of max_length to the batch
            # 3. add the last short shard if there is one to overflow_buffer
            input_ids_slices = [input_ids[i : i + max_length] for i in range(len(input_ids))[::max_length]]
            for input_ids_slice in input_ids_slices:
                if len(input_ids_slice) == max_length:
                    add_sample_to_batch(build_sample([input_ids_slice]))
                else:
                    add_to_overflow_buffer(input_ids_slice)
        else:
            # if input_ids < max_length add it to overflow_buffer
            add_to_overflow_buffer(input_ids)

        # at this point overflow_buffer may contain enough data for at most one more max_length sample
        if sum(len(x) for x in overflow_buffer) >= max_length:
            chunked_sample = []
            chunked_sample_len = 0
            for i in range(len(overflow_buffer)):
                chunk = overflow_buffer.pop(0)
                current_sample_len = chunked_sample_len + len(chunk)
                if current_sample_len <= max_length:
                    chunked_sample.append(chunk)
                    chunked_sample_len = current_sample_len
                else:
                    split_point = len(chunk) - (current_sample_len - max_length)
                    chunk_use, chunk_overflow = chunk[:split_point], chunk[split_point:]
                    chunked_sample.append(chunk_use)
                    overflow_buffer.insert(0, chunk_overflow)

                    break
            add_sample_to_batch(build_sample(chunked_sample))

    return packed_batch


class CausalDataConfig(DataConfig):
    pass


def slice_and_pack_dataset(self, dataset: DatasetType) -> DatasetType:
    """slice long sequences into chunks, then pack and pad any short cuts"""

    batch_size = min(100, len(dataset) // self.config.num_proc + 1)

    dataset = dataset.shuffle(seed=self.config.seed)

    overflow_buffer: List[List[int]] = []

    dataset = dataset.map(
        lambda x: slice_and_pack_causal_batch(
            x,
            max_length=self.config.max_length,
            overflow_buffer=overflow_buffer,
        ),
        batched=True,
        batch_size=batch_size,
        num_proc=self.config.num_proc,
        desc="Packing dataset",
    )
    if len(dataset) < 1:
        raise ValueError(f"No data left after packing dataset samples in {self.__class__.__name__}")
    return dataset


class CausalDataFactory(DataFactory):
    name = "causal"
    config: CausalDataConfig
    callbacks = [
        ("post-load", slice_and_pack_dataset),
    ]

    def process(self, dataset: DatasetType) -> DatasetType:
        if "text" not in dataset.column_names:
            raise ValueError("Dataset must have 'text' column to tokenize for CausalDataFactory.")
        dataset = dataset.select_columns(["text"])

        # normal tokenization
        return dataset.map(
            lambda example: {
                **self.tokenize_text(
                    example["text"],
                    self.tokenizer,
                )
            },
            remove_columns=dataset.column_names,
            num_proc=self.config.num_proc,
            desc="Tokenizing text",
        )

    @classmethod
    def tokenize_text(
        cls,
        text: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizerBase,
    ) -> BatchEncoding:
        # - we only want input_ids, as we rely on position_ids and not the attention_mask
        # - verbose=False because we potentially tokenize much longer sequences than the model can handle because later we slice the outcome into something that a model can handle, therefore tokenizer warnings like "Token indices sequence length is longer than the specified maximum sequence length" are irrelevant.
        return tokenizer(text, return_attention_mask=False, add_special_tokens=False, verbose=False)

    def create_dataloader(self, dataset: DatasetType) -> DataLoader:
        return DataLoader(
            dataset,
            collate_fn=DataCollatorForCausalLM(tokenizer=self.tokenizer, config=self.config),
            batch_size=self.micro_batch_size,
            sampler=DistributedSampler(dataset, num_replicas=self.world_size, rank=self.global_rank),
            num_workers=self.config.dl_num_workers,
            drop_last=True,
        )
