import pytest
import yaml

from arctic_training.config.trainer import get_config


def models_are_equal(model_a, model_b):
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        if not param_a.data.eq(param_b.data).all():
            return False

    return True


@pytest.mark.cpu
def test_ds_engine(tmp_path):
    config_dict = {
        "type": "sft",
        "exit_iteration": 2,
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
            "type": "deepspeed",
            "auto_resume": True,
            "output_dir": str(tmp_path / "checkpoints"),
            "save_end_of_training": True,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        f.write(yaml.dump(config_dict))

    config = get_config(config_path)
    trainer = config.trainer

    # Force checkpoint to be saved despite no training happening
    trainer.training_finished = True
    trainer.checkpoint()

    original_model = trainer.model
    config_dict["seed"] = 0  # Make sure newly initialized model is different
    with open(config_path, "w") as f:
        f.write(yaml.dump(config_dict))

    config = get_config(config_path)
    trainer = config.trainer
    loaded_model = trainer.model

    assert models_are_equal(original_model, loaded_model), "Models are not equal"
    # TODO: Add assertion on optimizer state
