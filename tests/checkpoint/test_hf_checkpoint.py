import os

import pytest
import yaml
from deepspeed.comm import init_distributed

from arctic_training.config.trainer import get_config


@pytest.mark.cpu
def test_hf_engine(tmp_path):
    os.environ["DS_ACCELERATOR"] = "cpu"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_SIZE"] = "1"
    init_distributed(auto_mpi_discovery=False)

    config_dict = {
        "type": "sft",
        "model": {
            "type": "random-weight-hf",
            "name_or_path": "HuggingFaceTB/SmolLM-135M-Instruct",
            "attn_implementation": "eager",
            "dtype": "float32",
        },
        "data": {
            "type": "noop",
            "sources": [],
        },
        "optimizer": {
            "type": "cpu-adam",
        },
        "scheduler": {
            "type": "noop",
        },
        "checkpoint": {
            "type": "huggingface",
            "output_dir": str(tmp_path / "checkpoints"),
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write(yaml.dump(config_dict))

    config = get_config(config_path)
    trainer = config.trainer
    print(trainer.model)
