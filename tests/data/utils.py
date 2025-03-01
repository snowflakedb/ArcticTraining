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

from typing import List
from pathlib import Path
from arctic_training.data.sft_factory import SFTDataFactory
from arctic_training.data.sft_factory import SFTDataConfig
from arctic_training.config.tokenizer import TokenizerConfig
from typing import Optional
from types import SimpleNamespace
from transformers import AutoTokenizer

def create_sft_data_factory(model_name: str, sources: List[str], eval_sources: List[str] = [], cache_dir: Optional[Path] = None) -> SFTDataFactory:
    data_config = SFTDataConfig(
        type="sft",
        sources=sources,
        eval_sources=eval_sources,
        use_data_cache=cache_dir is not None,
        cache_dir=cache_dir,
    )
    tokenizer_config = TokenizerConfig(type="huggingface", name_or_path=model_name)

    dummy_trainer = SimpleNamespace(
        config=SimpleNamespace(
            micro_batch_size=1,
            data=data_config,
            tokenizer=tokenizer_config,
        ),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
    )

    data_factory = data_config.factory(trainer=dummy_trainer)
    return data_factory