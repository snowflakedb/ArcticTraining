import argparse
from datasets import load_from_disk
from data_generation import construct_gpt_prompt


def main():
    parser = argparse.ArgumentParser(description="verify the gpt result")
    parser.add_argument("--verify-path", help="Chain of thought file path.", type=str)
    parser.add_argument(
        "--output-path",
        type=str,
    )
    args = parser.parse_args()

    data_verify = load_from_disk(args.verify_path)

    new_data = []
    for row in data_verify:
        if len(row["correct_answers"]) > 0:
            new_messages = construct_gpt_prompt(row)
            if len(new_messages) > 2:
                new_messages = [new_messages[0], new_messages[-1]]
            new_messages.append({"role": "assistant", "content": row["correct_answers"][0]})
            new_data.append(new_messages)

    new_dataset = Dataset.from_dict(
        {
            "messages": [d for d in new_data],
        }
    )
    new_dataset.save_to_disk(args.output_path)


if __name__ == "__main__":
    main()