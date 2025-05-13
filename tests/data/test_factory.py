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


def test_min_iterations(model_name: str, tmp_path: Path):
    # TODO: Make this test use a dummy factory of the base class rather than SFTDataFactory
    data_factory = create_data_factory(
        model_name=model_name,
        data_config_kwargs=dict(
            type="sft",
            sources=["HuggingFaceH4/ultrachat_200k:train[:1]"],
            cache_dir=tmp_path,
        ),
    )
    data_factory.trainer.config.min_iterations = 20

    trainer_dataloader, _ = data_factory()

    assert len(trainer_dataloader) == 20, "Dataloader did not have the correct number of batches"
