import os
import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.mark.gpu
def test_sft_trainer(tmp_path):
    config_dict = {
        "type": "sft",
        "code": str(Path(__file__).parent / "trainer_test_helpers.py"),
        "exit_iteration": 2,
        "micro_batch_size": 1,
        "model": {
            "type": "random-weight-hf",
            "name_or_path": "HuggingFaceTB/SmolLM-135M-Instruct",
            "attn_implementation": "eager",
        },
        "data": {
            "max_length": 2048,
            "sources": ["HuggingFaceH4/ultrachat_200k-truncated"],
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write(yaml.dump(config_dict))

    result = subprocess.run(
        f"arctic_training {config_path}", shell=True, text=True, capture_output=True
    )

    if result.returncode != 0:
        print(result.stderr)
        pytest.fail("Training failed")


@pytest.mark.cpu
def test_sft_trainer_cpu(tmp_path):
    config_dict = {
        "type": "sft",
        "code": str(Path(__file__).parent / "trainer_test_helpers.py"),
        "exit_iteration": 2,
        "micro_batch_size": 1,
        "model": {
            "type": "random-weight-hf",
            "name_or_path": "HuggingFaceTB/SmolLM-135M-Instruct",
            "attn_implementation": "eager",
            "dtype": "float32",
        },
        "data": {
            "max_length": 2048,
            "sources": ["HuggingFaceH4/ultrachat_200k-truncated"],
        },
        "deepspeed": {
            "zero_optimization": {
                "stage": 0,
            },
        },
        "optimizer": {
            "type": "cpu-adam",
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write(yaml.dump(config_dict))

    env = os.environ.copy()
    env["DS_ACCELERATOR"] = "cpu"

    result = subprocess.run(
        f"arctic_training {config_path}",
        shell=True,
        text=True,
        capture_output=True,
        env=env,
    )

    if result.returncode != 0:
        print(result.stderr)
        pytest.fail("Training failed")
