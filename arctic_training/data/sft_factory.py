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

import sys
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from tqdm import tqdm
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerBase

from arctic_training.config.data import DataConfig
from arctic_training.data.factory import DataFactory
from arctic_training.data.utils import DatasetType

IGNORE_INDEX = -100


# this function is modified from TRL trl.trainer.utils.py
def pad(
    tensors: List[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    is_position_id: bool = False,
    divisible_by: int = 256,
    max_seq: Optional[int] = None,
    dim_to_pad: int = -1,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`List[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        is_position_id (`bool`):
            If it is position_id, we will use arange to generate the position id in order to avoid too much padding causes flash attn crash.
        divisible_by (`int`):
            The number that the length of the sequence should be divisible by.
        max_seq (`int`):
            The maximum length of the sequence. If it is not None, we will truncate the sequence to the maximum length or pad the sequence to the maximum length.
        dim_to_pad (`int`):
            The dimension to pad. Default is -1.
    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()
    if max_seq is not None:
        output_shape[dim_to_pad] = max_seq
    elif divisible_by is not None:
        output_shape[dim_to_pad] = (
            int(np.ceil(output_shape[dim_to_pad] / divisible_by)) * divisible_by
        )

    # Create an output tensor filled with the padding value
    # TODO: Likely for 2D position ids, this does not work. Need to revisit.
    if is_position_id:
        output = (
            torch.arange(
                output_shape[dim_to_pad],
                dtype=tensors[0].dtype,
                device=tensors[0].device,
            )
            .repeat(len(tensors) * np.prod(output_shape) // output_shape[dim_to_pad])
            .view(len(tensors), *output_shape)
        )
    else:
        output = torch.full(
            (len(tensors), *output_shape),
            padding_value,
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")
        # import pdb; pdb.set_trace()
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t
    return output


class DataCollatorForCausalLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(example["input_ids"]) for example in instances]
        labels = [torch.tensor(example["labels"]) for example in instances]
        # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flash_attention_utils.py#L270
        # we do not need attention_mask when pos-id is provided and multi-seq packed
        # attention_mask = [
        #     torch.tensor(example["attention_mask"]) for example in instances
        # ]
        if "position_ids" in instances[0]:
            position_ids = [
                torch.tensor(example["position_ids"]) for example in instances
            ]
        else:
            position_ids = [
                torch.tensor(list(range(len(example["input_ids"]))))
                for example in instances
            ]

        input_ids = pad(input_ids, padding_value=self.tokenizer.pad_token_id)
        labels = pad(labels, padding_value=IGNORE_INDEX)
        position_ids = pad(position_ids, padding_value=0, is_position_id=True)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }


def packing_sft_dataset(
    dataset: DatasetType,
    seed: int,
    rank: int,
    max_length: int,
    always_max_length: bool,
) -> DatasetType:
    # packing for sft / cpt are different
    dataset = dataset.shuffle(seed=seed + rank)
    ds_keys = ("input_ids", "labels", "position_ids", "attention_mask")
    train_dataset: Dict[str, List] = {key: [] for key in ds_keys}
    example: Dict[str, List] = {key: [] for key in ds_keys}

    # pack multiple samples into one sample
    # for data in dataset:
    # TODO: make it multi-process?
    for data in tqdm(
        dataset,
        total=len(dataset),
        dynamic_ncols=True,
        file=sys.stdout,
        desc="Packing data",
        disable=rank != 0,
    ):
        input_ids, attention_mask, labels = (
            data["input_ids"],
            data["attention_mask"],
            data["labels"],
        )

        if (
            not always_max_length
            and len(example["input_ids"]) + len(input_ids) > max_length
        ) or len(example["input_ids"]) > max_length:
            for key in train_dataset.keys():
                train_dataset[key].append(example[key])

            example = {key: [] for key in ds_keys}

        example["input_ids"].extend(input_ids)
        example["labels"].extend(labels)
        example["position_ids"].extend(list(range(len(input_ids))))
        example["attention_mask"].extend(attention_mask)

    # add the last example
    if example["input_ids"]:
        for key in train_dataset.keys():
            train_dataset[key].append(example[key])

    return Dataset.from_dict(train_dataset)


class SFTDataConfig(DataConfig):
    max_length: int = 8192
    """ Maximum length of the input sequence. """

    mask_inputs: bool = True
    """ Whether to mask the input sequence. """

    always_max_length: bool = False
    """
    If this is turned on, each batch will be filled up to the max length by
    appending samples until the total length matches the max length. It might
    cause the last sample to be truncated.
    """


def filter_dataset_length(self, dataset: DatasetType) -> DatasetType:
    dataset = dataset.filter(
        lambda x: len(x["input_ids"]) <= self.config.max_length,
        num_proc=self.config.num_proc,
        desc="Filtering dataset by max length",
    )
    if len(dataset) < 1:
        raise ValueError(
            f"No data left after filtering by max length {self.config.max_length} in"
            f" {self.__class__.__name__}. Consider increasing the `max_length`."
        )
    return dataset


def pack_dataset(self, dataset: DatasetType) -> DatasetType:
    dataset = packing_sft_dataset(
        dataset,
        seed=self.config.seed,
        rank=self.global_rank,
        max_length=self.config.max_length,
        always_max_length=self.config.always_max_length,
    )
    if len(dataset) < 1:
        raise ValueError(
            f"No data left after packing dataset samples in {self.__class__.__name__}"
        )
    return dataset


class SFTDataFactory(DataFactory):
    name = "sft"
    config: SFTDataConfig
    callbacks = [
        ("post-load", filter_dataset_length),
        ("post-load", pack_dataset),  # Debug
    ]

    def process(self, dataset: DatasetType) -> DatasetType:
        if "messages" not in dataset.column_names:
            raise ValueError(
                "Dataset must have 'messages' column to tokenize for SFTDataFactory."
            )
        dataset = dataset.select_columns(["messages"])
        # sft based tokenization,
        # we assume the messages are in the format of:
        # {'role': '...', 'content': '...'}
        # datasets = datasets.select(range(100, 1100))
        dataset = dataset.select(range(len(dataset)))
        # datasets.disable_caching()
        # tmp = tokenize_messages(datasets[0]["messages"][:2], tokenizer, mask_inputs=mask_inputs)
        # import pdb; pdb.set_trace()
        return dataset.map(
            lambda ex: {
                **self.tokenize_messages(
                    ex["messages"],
                    self.tokenizer,
                    mask_inputs=self.config.mask_inputs,
                )
            },
            num_proc=self.config.num_proc,
            desc="Tokenizing messages",
        )

    @classmethod
    def tokenize_messages(
        cls,
        messages: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
    ) -> BatchEncoding:
        conversation_text = tokenizer.apply_chat_template(
            conversation=messages, tokenize=False
        )
        conversation_ids = tokenizer(
            conversation_text,
            return_offsets_mapping=mask_inputs,
            add_special_tokens=False,
        )

        if mask_inputs:
            assistant_ranges = cls.get_assistant_start_end_indices(
                messages, conversation_text
            )
            # _ = get_assistant_start_end_indices(messages, conversation_text)
            labels = cls.get_masked_labels(conversation_ids, assistant_ranges)
            conversation_ids["labels"] = labels
            # compare_messages_with_labels(split_list_by_specific_num(conversation_ids["labels"]), messages, tokenizer)
            del conversation_ids["offset_mapping"]
        else:
            conversation_ids["labels"] = conversation_ids["input_ids"]

        return conversation_ids

    @staticmethod
    # this code is adpoted from https://github.com/huggingface/trl/issues/632 (user: Peter-Devine )
    def get_assistant_start_end_indices(
        messages: List[Dict[str, str]], conversation_text: str
    ) -> List[Tuple[int, int]]:
        return_indices = []
        for message in messages:
            if message["role"] == "assistant":
                message_text = message["content"]
                match_index = conversation_text.find(message_text)
                # start_indices.append(match_index)
                end_indices = match_index + len(message_text)
                return_indices.append((match_index, end_indices))
        return return_indices

    @staticmethod
    def get_masked_labels(
        conversation_ids: BatchEncoding, assistant_ranges: List[Tuple[int, int]]
    ) -> List[int]:
        pre_output = IGNORE_INDEX
        output = []

        for id_, (id_s, id_e) in list(
            zip(
                conversation_ids["input_ids"],
                conversation_ids["offset_mapping"],
            )
        ):
            if any(id_s >= s and id_e <= e for s, e in assistant_ranges):
                pre_output = id_
                output.append(id_)
            else:
                # the if-else here is to include the eos token in the loss.
                # for instance, the asistent answer is
                # <|assistant|> I am good <eos> <|user|> xxx
                #      -100     1 2   3     4     -100       -100
                # after the shift, input_ids = input_ids[:-1], labels = labels[1:]
                #        1      2 3   4     -100  -100
                # now the prediction is correct, and the model will be able to predict <eos> token
                if pre_output != IGNORE_INDEX:
                    pre_output = IGNORE_INDEX
                    output.append(id_)
                else:
                    pre_output = IGNORE_INDEX
                    output.append(IGNORE_INDEX)
        return output

    def create_dataloader(self, dataset: DatasetType) -> DataLoader:
        return DataLoader(
            dataset,
            collate_fn=DataCollatorForCausalLM(tokenizer=self.tokenizer),
            batch_size=self.micro_batch_size,
            sampler=RandomSampler(dataset),
            num_workers=self.config.num_proc,
            drop_last=True,
        )
