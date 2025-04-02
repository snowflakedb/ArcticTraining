# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from arctic_training.config.trainer import get_config
from arctic_training.registry import get_registered_trainer

from .utils import models_are_equal


def test_hf_engine(tmp_path, model_name):
    config_dict = {
        "type": "sft",
        "skip_validation": True,
        "train_log_iter_interval": 0,
        "model": {
            "type": "random-weight-hf",
            "name_or_path": model_name,
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
    trainer_cls = get_registered_trainer(config.type)
    trainer = trainer_cls(config)

    # Force checkpoint to be saved despite no training happening
    trainer.training_finished = True
    trainer.checkpoint()

    # Store original model for comparison later
    original_model = trainer.model

    config_dict["model"]["name_or_path"] = str(trainer.checkpoint_engines[0].checkpoint_dir)
    config = get_config(config_dict)
    trainer_cls = get_registered_trainer(config.type)
    trainer = trainer_cls(config)

    loaded_model = trainer.model
    assert models_are_equal(original_model, loaded_model), "Models are not equal"
