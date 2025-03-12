import os
from functools import partial
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Tuple

from datasets import Dataset
from datasets import load_dataset, load_from_disk
from transformers import BatchEncoding
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from arctic_training.config import DataConfig

from arctic_training.data.loader import DataSetLoader
from arctic_training.register import register_dataset

IGNORE_INDEX = -100


class SFTDataSetLoader(DataSetLoader):
    dataset_type = "sft"

    def tokenize_fn(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        data_config: "DataConfig",
    ) -> Dataset:
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
                    ex["messages"], tokenizer, mask_inputs=data_config.mask_inputs
                )
            },
            num_proc=data_config.num_proc,
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
            conversation_ids = tokenizer(
                conversation_text,
                return_offsets_mapping=mask_inputs,
                add_special_tokens=False,
            )
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
            zip(conversation_ids["input_ids"], conversation_ids["offset_mapping"])
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

@register_dataset
class PromptResponsePairs(SFTDataSetLoader):
    dataset_name = "PromptResponsePairs"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        assert self.location is not None, "This data type must be given a location"
        dataset = load_from_disk(self.location)
        formatted_dataset = dataset.map(
            partial(
                self.instruct_format_conversation,
                query_key="prompt",
                response_key="response",
                source_name="SyntheticPromptResponsePairs",
            )
        )
        return formatted_dataset.select_columns(["messages"])
    
    @staticmethod
    def instruct_format_conversation(example, query_key, response_key, source_name):
        conversation = [
            {"role": "user", "content": example[query_key]},
            {"role": "assistant", "content": example[response_key]},
        ]
        return {
            "source": source_name,
            "messages": conversation,
        }

@register_dataset
class NoPromptResponseOnly(SFTDataSetLoader):
    dataset_name = "NoPromptResponseOnly"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        assert self.location is not None, "This data type must be given a location"
        dataset = load_from_disk(self.location)
        formatted_dataset = dataset.map(
            partial(
                self.instruct_format_conversation,
                query_key="prompt",
                response_key="response",
                source_name="SyntheticNoPromptResponseOnlyPairs",
            )
        )
        return formatted_dataset.select_columns(["messages"])
    
    @staticmethod
    def instruct_format_conversation(example, query_key, response_key, source_name):
        conversation = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": example[response_key]},
        ]
        return {
            "source": source_name,
            "messages": conversation,
        }


@register_dataset
class UltraChat200K(SFTDataSetLoader):
    dataset_name = "HuggingFaceH4/ultrachat_200k"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        return load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="test_sft" if eval else "train_sft",
            num_proc=num_proc,
        ).select_columns(["messages"])


@register_dataset
class OpenHermes2_5(SFTDataSetLoader):
    dataset_name = "teknium/OpenHermes-2.5"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        # OpenHermes-2.5 does not have an evaluation set
        if eval:
            return None

        dataset = load_dataset("teknium/OpenHermes-2.5")["train"]

        def process_example(example):
            return {
                "messages": [
                    {
                        "content": message["value"],
                        "role": {
                            "system": "system",
                            "human": "user",
                            "gpt": "assistant",
                        }[message["from"]],
                    }
                    for message in example["conversations"]
                ]
            }

        return dataset.map(
            process_example, num_proc=num_proc, desc="Loading openhermes"
        )


@register_dataset
class SlimOrca(SFTDataSetLoader):
    dataset_name = "Open-Orca/SlimOrca"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        if eval:
            assert False, "Test split does not exist."
        dataset = load_dataset("Open-Orca/SlimOrca", split="train", num_proc=num_proc)

        def process_example(example):
            return {
                "messages": [
                    {
                        "content": message["value"],
                        "role": {
                            "system": "system",
                            "human": "user",
                            "gpt": "assistant",
                        }[message["from"]],
                    }
                    for message in example["conversations"]
                ]
            }

        return dataset.map(process_example, num_proc=num_proc, desc="Loading slim orca")


@register_dataset
class MetaMathQA(SFTDataSetLoader):
    dataset_name = "meta-math/MetaMathQA"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        if eval:
            assert False, "Test split does not exist."
        dataset = load_dataset("meta-math/MetaMathQA", split="train", num_proc=num_proc)
        # TODO: figure out if we need this
        # if sample:
        #    shuffled_dataset = dataset.shuffle(seed=42)
        #    dataset = shuffled_dataset.select(range(sample))
        formatted_dataset = dataset.map(
            partial(
                self.instruct_format_conversation,
                query_key="query",
                response_key="response",
                source_name="MetaMathQA",
            )
        )
        return formatted_dataset.select_columns(["messages"])

    @staticmethod
    def instruct_format_conversation(example, query_key, response_key, source_name):
        conversation = [
            {"role": "user", "content": example[query_key]},
            {"role": "assistant", "content": example[response_key]},
        ]
        return {
            "source": source_name,
            "messages": conversation,
        }


@register_dataset
class MagicoderOSSInstruct75k(SFTDataSetLoader):
    dataset_name = "ise-uiuc/Magicoder-OSS-Instruct-75K"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        if eval:
            assert False, "Test split does not exist."
        dataset = load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", num_proc=num_proc
        )
        # TODO
        # if sample:
        #    shuffled_dataset = dataset.shuffle(seed=42)
        #    dataset = shuffled_dataset.select(range(sample))

        formatted_dataset = dataset.map(
            partial(
                self.instruct_format_conversation,
                query_key="problem",
                response_key="solution",
                source_name="Magicoder",
            )
        )
        return formatted_dataset.select_columns(["messages"])

    @staticmethod
    def instruct_format_conversation(example, query_key, response_key, source_name):
        conversation = [
            {"role": "user", "content": example[query_key]},
            {"role": "assistant", "content": example[response_key]},
        ]
        return {
            "source": source_name,
            "messages": conversation,
        }


@register_dataset
class LMSysChat1M(SFTDataSetLoader):
    dataset_name = "lmsys/lmsys-chat-1m"

    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        if eval:
            assert False, "Test split does not exist."
        dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", num_proc=num_proc, token=os.environ.get("HF_TOKEN", None))
        # TODO
        # if sample:
        #    shuffled_dataset = dataset.shuffle(seed=42)
        #    dataset = shuffled_dataset.select(range(sample))
        formatted_dataset = dataset.map(
            partial(self.vicuna_format_conversation, source_name="LMSYS-CHAT-1M")
        )
        return formatted_dataset.select_columns(["messages"])

    @staticmethod
    def vicuna_format_conversation(example, source_name):
        messages = []
        for conv in example["conversation"]:
            messages.append({"role": conv["role"], "content": conv["content"]})
        return {
            "source": source_name,
            "messages": messages,
        }
