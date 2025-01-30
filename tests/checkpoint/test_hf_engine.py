import pytest
from utils import models_are_equal

from arctic_training.config.trainer import get_config


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

    config = get_config(config_dict)
    trainer = config.trainer

    # Force checkpoint to be saved despite no training happening
    trainer.training_finished = True
    trainer.checkpoint()

    # Store original model for comparison later
    original_model = trainer.model

    config_dict["model"]["name_or_path"] = str(
        trainer.checkpoint_engines[0].checkpoint_dir
    )
    config = get_config(config_dict)
    trainer = config.trainer

    loaded_model = trainer.model
    assert models_are_equal(original_model, loaded_model), "Models are not equal"
