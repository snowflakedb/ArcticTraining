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

import json
from pathlib import Path

from arctic_training.checkpoint.hf_engine import HFCheckpointEngine

from .core.biencoder_model import Biencoder


class BiencoderCheckpointEngine(HFCheckpointEngine):
    name = "biencoder"

    @property
    def biencoder_config_file(self) -> Path:
        return self.checkpoint_dir / "biencoder_config.json"

    def save(self, model: Biencoder) -> None:
        super().save(model.encoder)

        # Save biencoder configuration details to a second config JSON file.
        biencoder_config_json = json.dumps({"pooling": model.pooling}, indent=2)
        (self.biencoder_config_file).write_text(biencoder_config_json)

    def load(self, model) -> None:
        raise NotImplementedError
