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
from typing import Any
from typing import Dict

import deepspeed
import safetensors as sf
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from arctic_training.checkpoint.engine import CheckpointEngine

# number of parameters per checkpoint shard
SHARD_SIZE = 2e9


class HFCheckpointEngine(CheckpointEngine):
    name = "huggingface"

    def model_file(self, file_name=None) -> Path:
        if file_name:
            return self.checkpoint_dir / file_name
        return self.checkpoint_dir / "model.pt"

    @property
    def config_file(self) -> Path:
        return self.checkpoint_dir / "config.json"

    @property
    def generation_config_file(self) -> Path:
        return self.checkpoint_dir / "generation_config.json"

    @staticmethod
    def _get_param(param: Any) -> Any:
        if hasattr(param, "ds_id"):
            params_to_fetch = []
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                params_to_fetch = [param]
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=True):
                return param.data.cpu()
        else:
            return param.data.cpu()

    def _get_ckpt_count(self, model) -> int:
        so_far_params = 0
        ckpt_count = 0
        param_list = []
        for p in model.parameters():
            param_list.append(p)
            so_far_params += p.ds_numel
            if so_far_params > SHARD_SIZE:
                ckpt_count += 1
                param_list = []
                so_far_params = 0
        if len(param_list) > 0:
            ckpt_count += 1
        return ckpt_count

    def _save_z3_checkpoint(self, model) -> None:
        output_state_dict = {}
        sf_index_json: Dict = {"metadata": {}, "weight_map": {}}
        so_far_params = 0
        checkpoint_id = 1
        total_params = 0
        model_to_save = model.module if hasattr(model, "module") else model
        ckpt_count = self._get_ckpt_count(model)

        # For PEFT models, we assume that trainable params will fit into memory
        if self.trainer.config.model.peft_config is not None:
            for k, v in model_to_save.named_parameters():
                if v.requires_grad:
                    v_p = self._get_param(v)
                    if model.global_rank == 0:
                        output_state_dict[k] = v_p
            if model.global_rank == 0:
                model.save_pretrained(
                    self.checkpoint_dir,
                    state_dict=output_state_dict,
                    safe_serialization=True,
                    max_shard_size="4GB",
                )
            return

        for k, v in model_to_save.named_parameters():
            v_p = self._get_param(v)
            if model.global_rank == 0:
                output_state_dict[k] = v_p

                so_far_params += v_p.numel()
                total_params += v_p.numel()
                if so_far_params > SHARD_SIZE:
                    tmp_file_name = f"model-{checkpoint_id:05}-{ckpt_count:05}.safetensors"
                    sf.torch.save_file(
                        output_state_dict,
                        self.model_file(tmp_file_name),
                        metadata={"format": "pt"},
                    )
                    # update the index file
                    for k in output_state_dict.keys():
                        sf_index_json["weight_map"][k] = tmp_file_name

                    del output_state_dict
                    output_state_dict = {}
                    so_far_params = 0
                    checkpoint_id += 1

        # save the last checkpoint
        if len(output_state_dict) > 0 and model.global_rank == 0:
            tmp_file_name = f"model-{checkpoint_id:05}-{ckpt_count:05}.safetensors"
            sf.torch.save_file(
                output_state_dict,
                self.model_file(tmp_file_name),
                metadata={"format": "pt"},
            )
            # update the index file
            for k in output_state_dict.keys():
                sf_index_json["weight_map"][k] = tmp_file_name

        # save the index file using json
        if model.global_rank == 0:
            sf_index_json["metadata"]["total_size"] = int(total_params * 2)
            with open(self.model_file("model.safetensors.index.json"), "w") as f:
                json.dump(sf_index_json, f)

        # save config file, ensure model arch matches the model class name
        model_to_save.config.architectures = [model_to_save.__class__.__name__]
        model_to_save.config.to_json_file(self.config_file)

    def save(self, model) -> None:
        if self.trainer.config.zero_3_enabled:
            self._save_z3_checkpoint(model)

        elif self.global_rank == 0:
            model.save_pretrained(
                self.checkpoint_dir,
                safe_serialization=True,
                max_shard_size="4GB",
            )

        self.trainer.tokenizer.save_pretrained(self.checkpoint_dir)

    def load(self, model) -> None:
        raise NotImplementedError
