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
from typing import Any

import deepspeed
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from arctic_training.checkpoint import CheckpointEngine
from arctic_training.registry import register


# TODO: bring in changes from https://github.com/snowflakedb/ArcticTraining/pull/10/files
@register
class HFCheckpointEngine(CheckpointEngine):
    name = "huggingface"

    @property
    def config_file(self) -> Path:
        return self.checkpoint_dir / "config.json"

    # TODO: make this match main branch
    @property
    def model_file(self) -> Path:
        return self.checkpoint_dir / "model.pt"

    @staticmethod
    def _get_param(param: Any) -> Any:
        if hasattr(param, "ds_id"):
            params_to_fetch = []
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                params_to_fetch = [param]
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=True):
                return param.data.cpu()
        else:
            return param.cpu()

    def _save_z3_checkpoint(self, model) -> None:
        output_state_dict = {}
        for k, v in model.named_parameters():
            v_p = self._get_param(v)
            if self.global_rank == 0:
                output_state_dict[k] = v_p

        if self.global_rank == 0:
            torch.save(output_state_dict, self.model_file)

        del output_state_dict

    def save(self, model) -> None:
        if self.trainer.config.zero_3_enabled:  # TODO implement this
            self._save_z3_checkpoint(model)
        elif self.global_rank == 0:
            model.save_pretrained(
                self.checkpoint_dir,
                safe_serialization=True,
                max_shard_size="4GB",
            )
            model.config.to_json_file(self.config_file)
            self.trainer.tokenizer.save_pretrained(self.checkpoint_dir)

    def load(self, model) -> None:
        raise NotImplementedError
