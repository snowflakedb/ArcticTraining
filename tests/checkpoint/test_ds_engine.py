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

import pytest
from utils import models_are_equal

from arctic_training.config.trainer import get_config


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

    config = get_config(config_dict)
    trainer = config.trainer

    # Force checkpoint to be saved despite no training happening
    trainer.training_finished = True
    trainer.checkpoint()

    # Store original model for comparison later
    original_model = trainer.model

    config_dict["seed"] = 0  # Make sure newly initialized model is different
    config = get_config(config_dict)
    trainer = config.trainer

    loaded_model = trainer.model
    assert models_are_equal(original_model, loaded_model), "Models are not equal"
    # TODO: Add assertion on optimizer state
