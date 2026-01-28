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

"""Data factory for On-Policy Distillation.

This module provides a data factory that loads prompts for on-policy distillation
training. Unlike SFT which loads full conversations, this factory loads only the
prompts that the student model will use to generate trajectories.
"""

from typing import Dict
from typing import List

import torch
from pydantic import Field
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerBase

from arctic_training.config.data import DataConfig
from arctic_training.config.utils import HumanInt
from arctic_training.data.factory import DataFactory
from arctic_training.data.hf_instruct_source import HFDataSourceInstruct
from arctic_training.data.utils import DatasetType


class OnPolicyDistillationDataConfig(DataConfig):
    """Configuration for On-Policy Distillation data loading."""

    div_length: HumanInt = 256
    """The number that the length of the prompt sequence should be divisible by."""

    max_prompt_length: HumanInt = Field(default=0, ge=0)
    """
    Maximum length of the prompt. If 0, uses max_length.
    Prompts longer than this will be truncated.
    """

    filter_long_prompts: bool = True
    """Whether to filter out prompts longer than max_prompt_length."""

    include_system_prompt: bool = True
    """Whether to include system prompts in the tokenized prompt."""


def pad_prompts(
    tensors: List[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "left",
    divisible_by: int = 256,
) -> torch.Tensor:
    """Pad a list of 1D tensors to the same length.

    For generation, we typically want left-padding so that the generated
    tokens are appended to the right.

    Args:
        tensors: List of 1D tensors to pad
        padding_value: Value to use for padding
        padding_side: Side to pad on ('left' or 'right')
        divisible_by: Pad to length divisible by this value

    Returns:
        Padded tensor of shape (batch_size, padded_length)
    """
    import math

    max_len = max(t.size(0) for t in tensors)
    padded_len = math.ceil(max_len / divisible_by) * divisible_by

    output = torch.full(
        (len(tensors), padded_len),
        padding_value,
        dtype=tensors[0].dtype,
    )

    for i, t in enumerate(tensors):
        if padding_side == "left":
            output[i, padded_len - t.size(0) :] = t
        else:
            output[i, : t.size(0)] = t

    return output


class DataCollatorForOnPolicyDistillation:
    """Data collator for on-policy distillation training.

    This collator pads prompts with left-padding (for generation) and
    tracks the original prompt lengths for later use.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, config: OnPolicyDistillationDataConfig):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(example["input_ids"]) for example in instances]
        prompt_lengths = [len(example["input_ids"]) for example in instances]

        # Left-pad for generation
        padded_input_ids = pad_prompts(
            input_ids,
            padding_value=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            padding_side="left",
            divisible_by=self.config.div_length,
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (padded_input_ids != (self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)).long()

        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "prompt_lengths": torch.tensor(prompt_lengths),
        }


def filter_by_prompt_length(self, dataset: DatasetType) -> DatasetType:
    """Filter dataset to remove prompts longer than max_prompt_length."""
    if not self.config.filter_long_prompts:
        return dataset

    max_len = self.config.max_prompt_length or self.config.max_length

    dataset = dataset.filter(
        lambda x: len(x["input_ids"]) <= max_len,
        num_proc=self.config.num_proc,
        desc="Filtering prompts by max length",
    )

    if len(dataset) < 1:
        raise ValueError(
            f"No data left after filtering by max prompt length {max_len}. "
            "Consider increasing max_prompt_length or max_length."
        )

    return dataset


class OnPolicyDistillationDataFactory(DataFactory):
    """Data factory for On-Policy Distillation training.

    This factory loads prompts (without completions) for on-policy distillation.
    The prompts are tokenized and left-padded for efficient batch generation
    by the student model.

    Expected input format:
    - Dataset with 'messages' column containing conversation turns
    - Only user/system messages are used; assistant messages are ignored
    """

    name = "on_policy_distillation"
    config: OnPolicyDistillationDataConfig
    default_source_cls = HFDataSourceInstruct

    callbacks = [
        ("post-load", filter_by_prompt_length),
    ]

    def process(self, dataset: DatasetType) -> DatasetType:
        """Process the dataset by tokenizing prompts.

        Extracts the prompt (system + user messages) from each conversation
        and tokenizes it for generation.
        """
        if "messages" not in dataset.column_names:
            raise ValueError("Dataset must have 'messages' column for OnPolicyDistillationDataFactory.")

        dataset = dataset.select_columns(["messages"])

        return dataset.map(
            lambda ex: {
                **self.tokenize_prompt(
                    ex["messages"],
                    self.tokenizer,
                    include_system=self.config.include_system_prompt,
                )
            },
            remove_columns=dataset.column_names,
            num_proc=self.config.num_proc,
            desc="Tokenizing prompts",
        )

    @classmethod
    def tokenize_prompt(
        cls,
        messages: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        include_system: bool = True,
    ) -> BatchEncoding:
        """Tokenize the prompt portion of a conversation.

        Extracts system and user messages (excluding assistant responses)
        and tokenizes them using the chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            tokenizer: Tokenizer to use
            include_system: Whether to include system messages

        Returns:
            BatchEncoding with input_ids for the prompt
        """
        # Extract prompt messages (everything before the first assistant response)
        prompt_messages = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "assistant":
                break  # Stop at first assistant message
            if role == "system" and not include_system:
                continue
            prompt_messages.append(msg)

        # If no prompt messages found, use the first user message
        if not prompt_messages:
            for msg in messages:
                if msg.get("role") == "user":
                    prompt_messages = [msg]
                    break

        if not prompt_messages:
            raise ValueError("No prompt messages found in conversation")

        # Apply chat template to get the prompt text
        # add_generation_prompt=True adds the assistant prefix for generation
        prompt_text = tokenizer.apply_chat_template(
            conversation=prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize the prompt
        encoded = tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        return encoded

    @classmethod
    def extract_prompt_and_completion(
        cls,
        messages: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        include_system: bool = True,
    ) -> Dict[str, List[int]]:
        """Extract and tokenize both prompt and completion from messages.

        This is useful for evaluation or when you have ground-truth completions.

        Args:
            messages: Full conversation messages
            tokenizer: Tokenizer to use
            include_system: Whether to include system messages

        Returns:
            Dict with 'prompt_ids' and 'completion_ids'
        """
        prompt_messages = []
        completion_messages = []
        seen_assistant = False

        for msg in messages:
            role = msg.get("role", "")
            if role == "assistant":
                seen_assistant = True
                completion_messages.append(msg)
            elif not seen_assistant:
                if role == "system" and not include_system:
                    continue
                prompt_messages.append(msg)

        # Tokenize prompt
        prompt_text = tokenizer.apply_chat_template(
            conversation=prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # Tokenize completion (if any)
        completion_ids = []
        if completion_messages:
            # Get full conversation and subtract prompt
            full_text = tokenizer.apply_chat_template(
                conversation=prompt_messages + completion_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            completion_ids = full_ids[len(prompt_ids) :]

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
        }

    def create_dataloader(self, dataset: DatasetType) -> DataLoader:
        """Create a DataLoader with the appropriate collator."""
        dataloader = super().create_dataloader(dataset)
        dataloader.collate_fn = DataCollatorForOnPolicyDistillation(
            tokenizer=self.tokenizer,
            config=self.config,
        )
        return dataloader
