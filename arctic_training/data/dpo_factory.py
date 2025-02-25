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
from arctic_training.registry import register
from arctic_training.data.sft_factory import pad

IGNORE_INDEX = -100

class DPODataConfig(DataConfig):
    max_length: int = 8192
    """ Maximum length of the input sequence. """

    max_prompt_length: int = 4096
    """ Maximum prompt length of the input sequence. """

    mask_inputs: bool = True
    """ Whether to mask the input sequence. """

    always_max_length: bool = False
    """
    If this is turned on, each batch will be filled up to the max length by
    appending samples until the total length matches the max length. It might
    cause the last sample to be truncated.
    """
    dpo_prompt_truncation_mode: str = "keep_start"

def _adjust_prompt_length(
    prompt_token,
    chosen_token,
    rejected_token,
) -> List[int]:
    c_len = len(chosen_token["prompt_input_ids"])
    r_len = len(rejected_token["prompt_input_ids"])
    min_len = min(c_len, r_len)

    for k, v in prompt_token.items():
        prompt_token[k] = v[:min_len]

    num_diff_tokens = sum(
        [
            a != b
            for a, b in zip(
                chosen_token["prompt_input_ids"], rejected_token["prompt_input_ids"]
            )
        ]
    )
    num_diff_len = abs(c_len - r_len)
    if num_diff_tokens > 1 or num_diff_len > 1:
        raise ValueError(
            "Chosen and rejected prompt_input_ids might only differ on the last token due to tokenizer merge ops."
        )

    return min_len




def add_bos_token_if_needed(
    bos_token_id,
    prompt_len_input_ids,
    prompt_tokens,
    chosen_prompt_len_input_ids,
    chosen_tokens,
    rejected_prompt_len_input_ids,
    rejected_tokens,
) -> Tuple:
    if bos_token_id is not None:
        if (
            prompt_len_input_ids == 0
            or bos_token_id != prompt_tokens["prompt_input_ids"][0]
        ):
            prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens[
                "prompt_input_ids"
            ]
            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens[
                "prompt_attention_mask"
            ]
        if (
            chosen_prompt_len_input_ids == 0
            or bos_token_id != chosen_tokens["prompt_input_ids"][0]
        ):
            chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens[
                "prompt_input_ids"
            ]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens[
                "prompt_attention_mask"
            ]
        if (
            rejected_prompt_len_input_ids == 0
            or bos_token_id != rejected_tokens["prompt_input_ids"][0]
        ):
            rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens[
                "prompt_input_ids"
            ]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens[
                "prompt_attention_mask"
            ]
    return prompt_tokens, chosen_tokens, rejected_tokens

def _build_sequence_tokens(tokens, prefix: str) -> None:
    sequence_tokens = {
        f"{prefix}_{k}": tokens[f"prompt_{k}"] + tokens[k]
        for k in ["input_ids", "attention_mask"]
    }
    sequence_tokens[f"{prefix}_labels"] = sequence_tokens[f"{prefix}_input_ids"][:]
    sequence_tokens[f"{prefix}_labels"][: len(tokens["prompt_input_ids"])] = [
        IGNORE_INDEX
    ] * len(tokens["prompt_input_ids"])
    return sequence_tokens


class DataCollatorForPref:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # instances is a list of dictionaries, each dictionary contains:
        # ['chosen', 'rejected', 'prompt',
        # 'chosen_input_ids', 'chosen_attention_mask', 'chosen_labels',
        # 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels',
        # 'prompt_input_ids', 'prompt_attention_mask']

        prompt_text = [example["prompt_text"] for example in instances]
        chosen_text = [example["chosen_text"] for example in instances]
        rejected_text = [example["rejected_text"] for example in instances]

        input_ids = [
            torch.tensor(example["chosen_input_ids"]) for example in instances
        ] + [torch.tensor(example["rejected_input_ids"]) for example in instances]
        labels = [torch.tensor(example["chosen_labels"]) for example in instances] + [
            torch.tensor(example["rejected_labels"]) for example in instances
        ]
        attention_mask = [
            torch.tensor(example["chosen_attention_mask"]) for example in instances
        ] + [torch.tensor(example["rejected_attention_mask"]) for example in instances]

        input_ids = pad(input_ids, padding_value=self.tokenizer.pad_token_id)
        labels = pad(labels, padding_value=IGNORE_INDEX)
        attention_mask = pad(attention_mask, padding_value=0)

        rt = {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "prompt": [example["prompt_input_ids"] for example in instances]
                "prompt_text": prompt_text,
                "chosen_text": chosen_text,
                "rejected_text": rejected_text,
            }
        return rt


@register
class DPODataFactory(DataFactory):
    name = "dpo"
    config: DPODataConfig

    def convert_text(self, tokenizer, conversations: List[Dict[str, str]]):
        chosen_text = tokenizer.apply_chat_template(
            conversation=conversations, tokenize=False
        )
        return chosen_text

    def process(self, dataset: DatasetType) -> DatasetType:
        if "prompt" not in dataset.column_names:
            raise ValueError(
                "Dataset must have 'prompt' column to tokenize for SFTDataFactory."
            )
        if "chosen" not in dataset.column_names:
            raise ValueError(
                "Dataset must have 'chosen' column to tokenize for SFTDataFactory."
            )
        if "rejected" not in dataset.column_names:
            raise ValueError(
                "Dataset must have 'rejected' column to tokenize for SFTDataFactory."
            )
        dataset = dataset.select_columns(["prompt", "chosen", "rejected"])
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
                    ex["prompt"],
                    ex["chosen"],
                    ex["rejected"],
                    self.tokenizer,
                    mask_inputs=self.config.mask_inputs,
                )
            },
            num_proc=self.config.num_proc,
            desc="Tokenizing messages",
        )



    def process_prompt(self, tokenizer, prompt_text: str):
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False
        )
        return {f"prompt_{k}": v for k, v in prompt_ids.items()}

    def process_answer(self, tokenizer, prompt, answer):
        full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
        prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
        prompt_input_ids = prompt_tokenized["input_ids"]
        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][
            len(prompt_input_ids) :
        ]
        if len(full_tokenized["input_ids"]) != len(prompt_input_ids + answer_input_ids):
            raise ValueError(
                "Prompt input ids and answer input ids should have the same length."
            )

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if (
            prompt_input_ids
            != full_tokenized["input_ids"][:response_token_ids_start_idx]
        ):
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][
            :response_token_ids_start_idx
        ]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError(
                "Prompt input ids and attention mask should have the same length."
            )
        return {
                "prompt_input_ids": prompt_input_ids,
                "prompt_attention_mask": prompt_attention_mask,
                "input_ids": answer_input_ids,
                "attention_mask": answer_attention_mask,
            }

    def _truncate_tokens(
        self,
        chosen_tokens,
        rejected_tokens,
        prompt_tokens,
    ) -> None:
        """
        Truncates the tokens in chosen, rejected, and prompt sequences to ensure they fit within the maximum length constraints.
        """
        if self.config.dpo_prompt_truncation_mode not in ["keep_start", "keep_end"]:
            raise ValueError(f"Invalid truncation mode: {self.config.dpo_prompt_truncation_mode}")

        longer_response_length = max(
            len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
        )

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if (
                len(answer_tokens["prompt_input_ids"]) + longer_response_length
                > self.config.max_length
            ):
                if self.config.dpo_prompt_truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.config.max_prompt_length]
                elif self.config.dpo_prompt_truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.config.max_prompt_length :]

        # if that's still too long, truncate the response from the end
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if (
                len(answer_tokens["prompt_input_ids"]) + longer_response_length
                > self.config.max_length
            ):
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][
                        : self.config.max_length - self.config.max_prompt_length
                    ]

        return chosen_tokens, rejected_tokens, prompt_tokens




    @classmethod
    def tokenize_messages(
        cls,
        prompt: List[Dict[str, str]],
        chosen: List[Dict[str, str]],
        rejected: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        mask_inputs: bool = True,
    ) -> BatchEncoding:
        """
        Args:
            prompt (List[Dict[str, str]]):
                Conversation List of Dict with keys as content and role.
                May include system round and user round.
            chosen (List[Dict[str, str]]):
                Conversation List of Dict with keys as content and role.
            rejected (List[Dict[str, str]]):
                Conversation List of Dict with keys as content and role.
            tokenizer (PreTrainedTokenizerBase):
                tokenizer to tokenize text
            mask_inputs (Bool):
                boolean value
        """

        prompt_text = self.convert_text(tokenizer, prompt)
        chosen_text_full = self.convert_text(tokenizer, prompt+chosen)
        rejcted_text_full =self.convert_text(tokenizer, prompt+rejected)
        chosen_text = chosen_text_full[len(prompt_text):]
        reject_text = rejcted_text_full[len(prompt_text):]

        # Some tokenizer may merge the end of end and start of the answer
        # It will make inconsistant between chosen and rejected prompt part
        prompt_tokens = self.process_prompt(tokenizer, prompt)
        chosen_tokens = self.process_answer(tokenizer, prompt_text, chosen_text)
        rejected_tokens = self.process_answer(tokenizer, prompt_text, reject_text)

        prompt_len_input_ids = _adjust_prompt_length(
            prompt_tokens, chosen_tokens, rejected_tokens
        )

        prompt_tokens, chosen_tokens, rejected_tokens = add_bos_token_if_needed(
            tokenizer.bos_token_id, prompt_len_input_ids,
            prompt_tokens, len(chosen_tokens["prompt_input_ids"]), chosen_tokens,
            len(rejected_tokens["prompt_input_ids"]), rejected_tokens,
        )


        chosen_tokens, rejected_tokens, prompt_tokens = self._truncate_tokens(
            chosen_tokens, rejected_tokens, prompt_tokens
        )
        chosen_tokens = _build_sequence_tokens(chosen_tokens, "chosen")
        rejected_tokens = _build_sequence_tokens(rejected_tokens, "rejected")

        batch = {}
        for data in [prompt_tokens, chosen_tokens, rejected_tokens]:
            for k, v in data.items():
                batch[k] = v
        batch['prompt_text'] = prompt_text
        batch['chosen_text'] = chosen_text
        batch['reject_text'] = reject_text

        return batch

    def create_dataloader(self, dataset: DatasetType) -> DataLoader:
        return DataLoader(
            dataset,
            collate_fn=DataCollatorForPref(tokenizer=self.tokenizer),
            batch_size=self.micro_batch_size,
            sampler=RandomSampler(dataset),
            num_workers=self.config.num_proc,
            drop_last=True,
        )
