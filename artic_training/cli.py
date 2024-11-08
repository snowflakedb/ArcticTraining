import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="ArticTraining config yaml file.")
    args, deepspeed_args = parser.parse_known_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")

    subprocess.run(
        [
            "deepspeed",
            *deepspeed_args,
            "arctic_training_run",
            "--config",
            str(args.config),
        ]
    )


def run_script():
    from arctic_training import get_config
    from arctic_training import trainer_factory

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, required=True, help="ArticTraining config to run."
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")

    config = get_config(args.config)
    trainer = trainer_factory(config)
    trainer.train()
