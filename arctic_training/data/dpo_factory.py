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

from typing import Dict
from typing import List
from typing import Literal
from typing import Tuple
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerBase

from arctic_training.config.data import DataConfig
from arctic_training.config.utils import HumanInt
from arctic_training.data.factory import DataFactory
from arctic_training.data.sft_factory import pad
from arctic_training.data.utils import DatasetType

IGNORE_INDEX = -100


class DPODataConfig(DataConfig):
    max_prompt_length: HumanInt = 4096
    """ Maximum prompt length of the input sequence. """

    dpo_prompt_truncation_mode: Literal["keep_start", "keep_end"] = "keep_start"
    """
    Truncation mode to use when the sequence exceeds max_length. Possible values are "keep_end" and "keep_start".
    """


def _adjust_prompt_length(
    prompt_token: Dict[str, torch.Tensor],
    chosen_token: Dict[str, torch.Tensor],
    rejected_token: Dict[str, torch.Tensor],
) -> None:
    c_len = len(chosen_token["prompt_input_ids"])
    r_len = len(rejected_token["prompt_input_ids"])
    min_len = min(c_len, r_len)

    for k, v in prompt_token.items():
        prompt_token[k] = v[:min_len]

    num_diff_tokens = sum(
        [a != b for a, b in zip(chosen_token["prompt_input_ids"], rejected_token["prompt_input_ids"])]
    )
    num_diff_len = abs(c_len - r_len)
    if num_diff_tokens > 1 or num_diff_len > 1:
        raise ValueError(
            "Chosen and rejected prompt_input_ids might only differ on the last token due to tokenizer merge ops."
        )


def add_bos_token_if_needed(
    bos_token_id: Union[None, int],
    tokens: Dict[str, List[int]],
) -> Dict[str, List[int]]:
    if bos_token_id is None:
        return tokens

    len_input_ids = len(tokens["prompt_input_ids"])
    if (len_input_ids == 0) or (bos_token_id != tokens["prompt_input_ids"][0]):
        tokens["prompt_input_ids"].insert(0, bos_token_id)
        tokens["prompt_attention_mask"].insert(0, 1)
    return tokens


def add_eos_token_if_needed(
    eos_token_id: Union[None, int],
    tokens: Dict[str, List[int]],
) -> Dict[str, List[int]]:
    if eos_token_id is None:
        return tokens
    if len(tokens["input_ids"]) == 0 or eos_token_id != tokens["input_ids"][-1]:
        if len(tokens["input_ids"]) > 0 and eos_token_id == tokens["input_ids"][-2]:
            tokens["input_ids"] = tokens["input_ids"][:-1]
            tokens["attention_mask"] = tokens["attention_mask"][:-1]
        else:
            tokens["input_ids"].append(eos_token_id)
            tokens["attention_mask"].append(1)
    return tokens


def _build_sequence_tokens(tokens: Dict[str, List[int]], prefix: str) -> Dict[str, List[int]]:
    sequence_tokens = {f"{prefix}_{k}": tokens[f"prompt_{k}"] + tokens[k] for k in ["input_ids", "attention_mask"]}
    sequence_tokens[f"{prefix}_labels"] = sequence_tokens[f"{prefix}_input_ids"][:]
    sequence_tokens[f"{prefix}_labels"][: len(tokens["prompt_input_ids"])] = [IGNORE_INDEX] * len(
        tokens["prompt_input_ids"]
    )
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

        input_ids = [torch.tensor(example["chosen_input_ids"]) for example in instances] + [
            torch.tensor(example["rejected_input_ids"]) for example in instances
        ]
        labels = [torch.tensor(example["chosen_labels"]) for example in instances] + [
            torch.tensor(example["rejected_labels"]) for example in instances
        ]
        attention_mask = [torch.tensor(example["chosen_attention_mask"]) for example in instances] + [
            torch.tensor(example["rejected_attention_mask"]) for example in instances
        ]

        input_ids = pad(input_ids, padding_value=self.tokenizer.pad_token_id)
        labels = pad(labels, padding_value=IGNORE_INDEX)
        attention_mask = pad(attention_mask, padding_value=0)

        rt = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prompt": [example["prompt_input_ids"] for example in instances],
            "prompt_text": prompt_text,
            "chosen_text": chosen_text,
            "rejected_text": rejected_text,
        }
        return rt


class DPODataFactory(DataFactory):
    name = "dpo"
    config: DPODataConfig

    def convert_text(self, tokenizer, conversations: List[Dict[str, str]]) -> str:
        chosen_text = tokenizer.apply_chat_template(conversation=conversations, tokenize=False)
        return chosen_text

    def process(self, dataset: DatasetType) -> DatasetType:
        missing_columns = [c for c in ("prompt", "chosen", "rejected") if c not in dataset.column_names]
        if len(missing_columns) > 0:
            raise ValueError(
                "Dataset must have 'prompt', 'chosen', and 'rejected' columns to"
                " tokenizer for DPODataFactory. Missing the following columns:"
                f" {missing_columns}"
            )
        dataset = dataset.select_columns(["prompt", "chosen", "rejected"])

        return dataset.map(
            lambda ex: {
                **self.tokenize_messages(
                    ex["prompt"],
                    ex["chosen"],
                    ex["rejected"],
                    self.tokenizer,
                )
            },
            num_proc=self.config.num_proc,
            desc="Tokenizing messages",
        )

    def process_prompt(self, tokenizer, prompt_text: str) -> Dict[str, List[int]]:
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)
        return {f"prompt_{k}": v for k, v in prompt_ids.items()}

    def process_answer(self, tokenizer, prompt, answer) -> Dict[str, List[int]]:
        full_tokenized = tokenizer(prompt + answer, add_special_tokens=False)
        prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
        prompt_input_ids = prompt_tokenized["input_ids"]
        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]
        if len(full_tokenized["input_ids"]) != len(prompt_input_ids + answer_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")
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
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
        """
        Truncates the tokens in chosen, rejected, and prompt sequences to ensure they fit within the maximum length constraints.
        """
        if self.config.dpo_prompt_truncation_mode not in ["keep_start", "keep_end"]:
            raise ValueError(f"Invalid truncation mode: {self.config.dpo_prompt_truncation_mode}")

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.config.max_length:
                if self.config.dpo_prompt_truncation_mode == "keep_start":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.config.max_prompt_length]
                elif self.config.dpo_prompt_truncation_mode == "keep_end":
                    for k in ["prompt_input_ids", "prompt_attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][-self.config.max_prompt_length :]

        # if that's still too long, truncate the response from the end
        for answer_tokens in [chosen_tokens, rejected_tokens]:
            if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.config.max_length:
                for k in ["input_ids", "attention_mask"]:
                    answer_tokens[k] = answer_tokens[k][: self.config.max_length - self.config.max_prompt_length]

        return chosen_tokens, rejected_tokens, prompt_tokens

    def tokenize_messages(
        self,
        prompt: List[Dict[str, str]],
        chosen: List[Dict[str, str]],
        rejected: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizerBase,
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
        chosen_text_full = self.convert_text(tokenizer, prompt + chosen)
        rejcted_text_full = self.convert_text(tokenizer, prompt + rejected)
        chosen_text = chosen_text_full[len(prompt_text) :]
        reject_text = rejcted_text_full[len(prompt_text) :]

        # Some tokenizer may merge the end of end and start of the answer
        # It will make inconsistant between chosen and rejected prompt part
        prompt_tokens = self.process_prompt(tokenizer, prompt_text)
        chosen_tokens = self.process_answer(tokenizer, prompt_text, chosen_text)
        rejected_tokens = self.process_answer(tokenizer, prompt_text, reject_text)

        # This is being dropped on the floor - problem?
        _adjust_prompt_length(prompt_tokens, chosen_tokens, rejected_tokens)

        prompt_tokens = add_bos_token_if_needed(tokenizer.bos_token_id, prompt_tokens)
        chosen_tokens = add_bos_token_if_needed(tokenizer.bos_token_id, chosen_tokens)
        rejected_tokens = add_bos_token_if_needed(tokenizer.bos_token_id, rejected_tokens)
        chosen_tokens = add_eos_token_if_needed(tokenizer.eos_token_id, chosen_tokens)
        rejected_tokens = add_eos_token_if_needed(tokenizer.eos_token_id, rejected_tokens)

        chosen_tokens, rejected_tokens, prompt_tokens = self._truncate_tokens(
            chosen_tokens, rejected_tokens, prompt_tokens
        )
        chosen_tokens = _build_sequence_tokens(chosen_tokens, "chosen")
        rejected_tokens = _build_sequence_tokens(rejected_tokens, "rejected")

        row: Dict[str, Union[List[int], str]] = {}
        for data in [prompt_tokens, chosen_tokens, rejected_tokens]:
            for k, v in data.items():
                row[k] = v
        row["prompt_text"] = prompt_text
        row["chosen_text"] = chosen_text
        row["rejected_text"] = reject_text

        return row

    def create_dataloader(self, dataset: DatasetType) -> DataLoader:
        return DataLoader(
            dataset,
            collate_fn=DataCollatorForPref(tokenizer=self.tokenizer),
            batch_size=self.micro_batch_size,
            sampler=DistributedSampler(dataset, num_replicas=self.world_size, rank=self.global_rank),
            num_workers=self.config.dl_num_workers,
            drop_last=True,
        )
