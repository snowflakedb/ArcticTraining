import os
import subprocess
from pathlib import Path

import pytest
import yaml


@pytest.mark.cpu
def test_trainer(tmp_path):
    config_dict = {
        "type": "sft",
        "code": str(Path(__file__).parent / "cpu_trainer.py"),
        "exit_iteration": 2,
        "micro_batch_size": 1,
        "model": {
            "name_or_path": "HuggingFaceTB/SmolLM-135M-Instruct",
            "attn_implementation": "eager",
            "dtype": "float32",
        },
        "data": {
            "max_length": 1024,
            "sources": ["ise-uiuc/Magicoder-OSS-Instruct-75K"],
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
