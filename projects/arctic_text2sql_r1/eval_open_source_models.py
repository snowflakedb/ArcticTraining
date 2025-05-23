import os
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Run bird eval over one or more models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model checkpoint directories to evaluate"
    )
    parser.add_argument(
        "--input_file",
        nargs="+",
        required=True,
        help="Input file for evaluation"
    )
    parser.add_argument(
        "--parallel-generation",
        action="store_true",
        help="If set, enables parallel generation mode"
    )
    parser.add_argument(
        "--gold_file_path",
        type=str,
        default="/data/bohan/bird_submission/bird/dev_20240627/dev.json",
        help="Path to the gold file for evaluation",
    )
    parser.add_argument(
        "--dp_path",
        type=str,
        default="/data/bohan/bird_submission/bird/dev_20240627/dev_databases",
        help="Path to the database for evaluation",
    )
    args = parser.parse_args()

    models = args.models
    PARALLEL_GENERATION = args.parallel_generation

    for model in models:
        # safe name for output dirs / eval names
        model_name = model.replace("/", "_")

        extra_param = ""
        if PARALLEL_GENERATION:
            visible_devices = "0,1,2,3,4,5,6,7"
            extra_param = "--parallel_generation"
        else:
            # pick devices based on model size tag
            if "7b" in model_name:
                visible_devices = "0,1,2,3"
            elif "14b" in model_name or "32b" in model_name:
                visible_devices = "0,1,2,3,4,5,6,7"
            else:
                # fallback if no size tag matched
                visible_devices = "0,1,2,3,4,5,6,7"

        tensor_parallel_size = len(visible_devices.split(","))

        dev_bird_eval_name = f"{model_name}_dev_bird"
        cmd = (
            f"python3 auto_evaluation.py {extra_param} "
            f"--output_ckpt_dir {model} "
            f"--source bird "
            f"--visible_devices {visible_devices} "
            f"--input_file {args.input_file} "
            f"--eval_name {dev_bird_eval_name} "
            f"--tensor_parallel_size {tensor_parallel_size} "
            f"--n 1 "
            f"--gold_file {args.gold_file_path} "
            f"--db_path {args.dp_path}"
        )

        print(f"Running:\n  {cmd}\n")
        os.system(cmd)


if __name__ == "__main__":
    main()