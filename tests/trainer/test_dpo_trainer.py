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


def test_dpo_trainer_cpu(model_name):
    config_dict = {
        "type": "dpo",
        "beta": 0.1,
        "skip_validation": True,
        "exit_iteration": 2,
        "micro_batch_size": 1,
        "model": {
            "type": "random-weight-hf",
            "name_or_path": model_name,
            "dtype": "float32",
        },
        "ref_model": {
            "type": "random-weight-hf",
            "name_or_path": model_name,
            "dtype": "float32",
        },
        "data": {
            "max_length": 2048,
            "sources": ["HuggingFaceH4/ultrafeedback_binarized:train[:20]"],
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

    config = get_config(config_dict)
    trainer_cls = get_registered_trainer(config.type)
    trainer = trainer_cls(config)
    trainer.train()
    assert trainer.global_step > 0, "Training did not run"
