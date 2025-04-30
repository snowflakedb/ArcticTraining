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

from types import SimpleNamespace

import pytest
from peft import LoraConfig
from peft import PeftConfig
from peft import PeftModel
from peft import VeraConfig

from arctic_training.config.model import ModelConfig


@pytest.mark.parametrize("peft_type, config_cls", [("Lora", LoraConfig), ("Vera", VeraConfig)])
def test_peft_config(model_name: str, peft_type: str, config_cls: PeftConfig):
    config_dict = {
        "type": "random-weight-hf",
        "name_or_path": model_name,
        "peft_config": {
            "peft_type": peft_type,
        },
    }
    config = ModelConfig(**config_dict)

    assert isinstance(
        config.peft_config_obj, config_cls
    ), f"Expected {config_cls} PEFT config type but got {type(config.peft_config)}"


def test_peft_config_fail(model_name: str):
    config_dict = {
        "type": "random-weight-hf",
        "name_or_path": model_name,
        "peft_config": {
            "peft_type": "invalid",
        },
    }
    with pytest.raises(ValueError):
        _ = ModelConfig(**config_dict)


def test_peft_model(model_name: str):
    config_dict = {
        "type": "random-weight-hf",
        "name_or_path": model_name,
        "peft_config": {
            "peft_type": "Lora",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    }
    config = ModelConfig(**config_dict)

    dummy_trainer = SimpleNamespace(config=SimpleNamespace(model=config))
    model_factory = config.factory(dummy_trainer)
    model = model_factory()

    assert isinstance(model, PeftModel), f"Expected PeftModel but got {type(model)}"
