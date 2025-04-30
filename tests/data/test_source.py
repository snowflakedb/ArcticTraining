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

from .utils import create_data_factory


def test_data_source_cache_path_uniqueness(model_name: str, tmp_path: Path):
    data_sources = [
        "HuggingFaceH4/ultrachat_200k",
        "Open-Orca/SlimOrca",
    ]
    data_factory = create_data_factory(
        model_name=model_name,
        data_config_kwargs=dict(
            type="sft",
            sources=data_sources,
            eval_sources=data_sources,
            cache_dir=tmp_path,
        ),
    )

    cache_paths = [s.cache_path for s in data_factory._get_data_sources(data_factory.config.sources)] + [
        s.cache_path for s in data_factory._get_data_sources(data_factory.config.eval_sources)
    ]
    assert len(cache_paths) == 2 * len(data_sources), "Cache paths were not generated for all data sources"
    assert len(cache_paths) == len(set(cache_paths)), "Cache paths were not unique"
