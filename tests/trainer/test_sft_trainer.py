import subprocess

import pytest
import yaml


@pytest.mark.gpu
def test_sft_trainer(tmp_path):
    config_dict = {
        "type": "sft",
        "model": {
            "name_or_path": "meta-llama/Llama-3.2-1B-Instruct",
            "attn_implementation": "eager",
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

    result = subprocess.run(
        f"arctic_training {config_path}", shell=True, text=True, capture_output=True
    )

    if result.returncode != 0:
        print(result.stderr)
        pytest.fail("Training failed")
