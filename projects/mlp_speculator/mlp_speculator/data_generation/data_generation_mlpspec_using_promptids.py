import argparse
import json
import os

from datasets import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM
from vllm import SamplingParams


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process UltraChat dataset configuration."
    )

    # Add command-line arguments
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="ultrachat",
        help="Path to your UltraChat dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Model name or path",
    )
    parser.add_argument(
        "--tensor_parallel",
        type=int,
        default=1,
        help="Number of tensor parallelism splits",
    )
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        default="test",
        help="Output path for the Hugging Face dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of prompts to process in each batch per node",
    )
    parser.add_argument(
        "--cur_split",
        type=int,
        default=0,
        help="Number of prompts to process in each batch per node",
    )
    parser.add_argument(
        "--total_split",
        type=int,
        default=32,
        help="Number of prompts to process in each batch per node",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Max tokens to generate"
    )
    parser.add_argument(
        "--gen_prompt_length", type=int, default=64, help="Max tokens to generate"
    )
    parser.add_argument(
        "--hostfile", type=str, default=None, help="Path to the hostfile"
    )
    parser.add_argument(
        "--skip_launch",
        action="store_true",  # This makes it a boolean flag
        help="Skip the launch process if this flag is set. Handy if you just want to run generation and server have been launched already",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",  # This makes it a boolean flag
        help="Skip the generation if this flag is set. Handy if you just want to launch the vllm servers.",
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


# Load dataset (Hugging Face format)
def load_hf_dataset(dataset):
    if dataset == "ultrachat":
        return load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            num_proc=32,
        )
    elif dataset == "magicoder":
        result = load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K",
            split="train",
            num_proc=32,
        )

        def instruct_format_conversation(example, query_key, response_key, source_name):
            conversation = [
                {"role": "user", "content": example[query_key]},
                {"role": "assistant", "content": example[response_key]},
            ]
            return {
                "source": source_name,
                "messages": conversation,
            }

        from functools import partial

        result = result.map(
            partial(
                instruct_format_conversation,
                query_key="problem",
                response_key="solution",
                source_name="Magicoder",
            )
        )
        return result

    else:
        print(f"Dataset {dataset} not supported")
        exit(0)


# Save responses as a Hugging Face dataset
def save_as_huggingface_dataset(prompts, responses, output_path):
    assert output_path is not None, "Please provide an output_path"
    data = [
        {"prompt": prompt, "response": response}
        for prompt, response in zip(prompts, responses)
    ]
    dataset = Dataset.from_dict(
        {
            "prompt": [d["prompt"] for d in data],
            "response": [d["response"] for d in data],
        }
    )
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")


def generate(args):
    # Load dataset
    os.makedirs(args.output_dataset_path, exist_ok=True)
    dataset = load_hf_dataset(args.hf_dataset)
    start = args.cur_split * len(dataset) // args.total_split
    end = (args.cur_split + 1) * len(dataset) // args.total_split
    dataset = dataset.select(range(start, end))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=1024,
        enable_chunked_prefill=True,
        # distributed_executor_backend="ray",
        gpu_memory_utilization=0.96,
    )

    sampling_params = SamplingParams(
        temperature=1,
        top_k=10,
        max_tokens=256,
        ignore_eos=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        detokenize=False,
    )
    gen_prompt_length = args.gen_prompt_length

    def preproc(d):
        message = d["messages"]
        tokenized_message = tokenizer.apply_chat_template(
            conversation=message, tokenize=False
        )
        tokenized_message = tokenizer(
            tokenized_message, add_special_tokens=False, return_tensors="pt"
        )
        input_ids = tokenized_message["input_ids"][0]
        max_len = len(input_ids) // gen_prompt_length * gen_prompt_length
        input_ids = input_ids[:max_len].reshape(-1, gen_prompt_length)
        d["messages"] = input_ids.tolist()
        return d

    dataset = dataset.map(preproc, num_proc=4)
    all_prompts = sum(dataset["messages"], [])

    outputs = llm.generate(
        sampling_params=sampling_params, prompt_token_ids=all_prompts
    )
    with open(
        f"{args.output_dataset_path}/{args.cur_split}_{args.total_split}.jsonl", "w"
    ) as f:
        for o in outputs:
            new_data = {"input": o.prompt_token_ids, "output": o.outputs[0].token_ids}
            f.write(json.dumps(new_data) + "\n")
            f.flush()


# Main function to process the dataset and generate responses in batches
def main():

    # parse arguments from command line
    args = parse_arguments()

    # Start the generation
    generate(args)


if __name__ == "__main__":
    main()
