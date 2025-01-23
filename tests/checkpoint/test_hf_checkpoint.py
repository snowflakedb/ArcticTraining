import pytest
import yaml

from arctic_training.config.trainer import get_config


def compare_model_weights(model_a, model_b):
    mismatched_params = []
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        if not param_a.data.eq(param_b.data).all():
            mismatched_params.append((param_a, param_b))
    return mismatched_params


@pytest.mark.cpu
def test_hf_engine(tmp_path):
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
    checkpoint_path = trainer.checkpoint_engines[0].checkpoint_dir
    original_model = trainer.model
    config_dict["model"]["name_or_path"] = str(checkpoint_path)
    with open(config_path, "w") as f:
        f.write(yaml.dump(config_dict))

    config = get_config(config_path)
    trainer = config.trainer
    loaded_model = trainer.model

    mismatched_params = compare_model_weights(original_model, loaded_model)
    assert not mismatched_params, f"mismatched params: {mismatched_params}"
