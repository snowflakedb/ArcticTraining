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
from typing import Dict

from transformers import AutoTokenizer

from arctic_training.config.tokenizer import TokenizerConfig
from arctic_training.data.factory import DataFactory
from arctic_training.registry import _get_class_attr_type_hints
from arctic_training.registry import get_registered_data_factory


def create_data_factory(
    model_name: str,
    data_config_kwargs: Dict,
) -> DataFactory:
    # Default to "sft" type if not provided
    data_config_kwargs["type"] = data_config_kwargs.get("type", "sft")

    factory_cls = get_registered_data_factory(data_config_kwargs["type"])
    data_config_cls = _get_class_attr_type_hints(factory_cls, "config")[0]
    data_config = data_config_cls(**data_config_kwargs)

    tokenizer_config = TokenizerConfig(type="huggingface", name_or_path=model_name)

    dummy_trainer = SimpleNamespace(
        config=SimpleNamespace(
            micro_batch_size=1,
            data=data_config,
            tokenizer=tokenizer_config,
            seed=42,
            gradient_accumulation_steps=1,
            min_iterations=0,
            train_log_iter_interval=0,
        ),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        _set_seeds=lambda seed: None,
    )

    data_factory = data_config.factory(trainer=dummy_trainer)
    return data_factory
