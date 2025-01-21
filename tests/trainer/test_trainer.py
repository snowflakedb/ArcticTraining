import os
import subprocess
from pathlib import Path

import pytest
import yaml


def test_trainer(tmp_path):
    config_dict = {
        "type": "dummy",
        "code": str(Path(__file__).parent / "dummy_trainer.py"),
        "model": {
            "name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
            "attn_implementation": "eager",
            "dtype": "float16",
        },
        "deepspeed": {
            "zero_optimization": {
                "stage": 0,
            },
        },
        "exit_iteration": 10,
        "micro_batch_size": 1,
        "data": {
            "sources": ["ise-uiuc/Magicoder-OSS-Instruct-75K"],
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write(yaml.dump(config_dict))

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

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
