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

from pathlib import Path
from types import SimpleNamespace

from arctic_training.config.tokenizer import TokenizerConfig
from arctic_training.data.sft_factory import SFTDataConfig


def test_cache_path_uniqueness(model_name: str, tmp_path: Path):
    data_sources = [
        "HuggingFaceH4/ultrachat_200k",
        "Open-Orca/SlimOrca",
        "meta-math/MetaMathQA",
        "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "lmsys/lmsys-chat-1m",
    ]
    data_config = SFTDataConfig(
        type="sft",
        sources=data_sources,
        eval_sources=data_sources,
        use_data_cache=True,
        cache_dir=tmp_path,
    )
    tokenizer_config = TokenizerConfig(type="huggingface", name_or_path=model_name)

    dummy_trainer = SimpleNamespace(
        config=SimpleNamespace(
            data=data_config,
            tokenizer=tokenizer_config,
        ),
    )

    data_factory = data_config.factory(trainer=dummy_trainer)
    cache_paths = [s.cache_path for s in data_factory._get_data_sources("train")] + [
        s.cache_path for s in data_factory._get_data_sources("eval")
    ]
    assert len(cache_paths) == 2 * len(
        data_sources
    ), "Cache paths were not generated for all data sources"
    assert len(cache_paths) == len(set(cache_paths)), "Cache paths were not unique"
