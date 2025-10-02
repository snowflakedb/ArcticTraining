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
from arctic_training import Trainer
from arctic_training.config.trainer import get_config
from arctic_training.registry import get_registered_trainer

_default_training_params = {
    "skip_validation": True,
    "exit_iteration": 2,
    "micro_batch_size": 1,
}

_cpu_training_params = {
    "deepspeed": {
        "zero_optimization": {
            "stage": 0,
        },
    },
    "optimizer": {
        "type": "cpu-adam",
    },
}


def run_dummy_training(config_dict, run_on_cpu: bool = True) -> "Trainer":
    """
    Trains a model based on the provided configuration dictionary.

    Args:
        config_dict (dict): Configuration dictionary for the training process.
        run_on_cpu (bool): If True, uses CPU-specific training parameters.

    Returns:
        Trainer: The trained trainer object.
    """
    default_training_params = _default_training_params
    if run_on_cpu:
        default_training_params |= _cpu_training_params

    config = get_config(default_training_params | config_dict)
    trainer_cls = get_registered_trainer(config.type)
    trainer = trainer_cls(config)
    trainer.train()

    assert trainer.global_step > 0, "Training did not run"

    return trainer
