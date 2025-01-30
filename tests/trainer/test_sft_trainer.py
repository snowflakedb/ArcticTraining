import pytest
import yaml

from arctic_training.config.trainer import get_config


@pytest.mark.gpu
def test_sft_trainer(tmp_path):
    config_dict = {
        "type": "sft",
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

    config = get_config(config_path)
    trainer = config.trainer
    trainer.train()
    assert trainer.global_step > 0, "Training did not run"


@pytest.mark.cpu
def test_sft_trainer_cpu(tmp_path):
    config_dict = {
        "type": "sft",
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

    config = get_config(config_path)
    trainer = config.trainer
    trainer.train()
    assert trainer.global_step > 0, "Training did not run"
