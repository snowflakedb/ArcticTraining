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

import copy
import os

import torch
import torch.distributed as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from arctic_training.checkpoint import CheckpointEngine
from arctic_training.logging import logger
from arctic_training.register import register_checkpoint

from .speculator import MLPSpeculator


@register_checkpoint
class MLPSpeculatorCheckpointEngine(CheckpointEngine):
    checkpoint_type: str = "mlp_speculator"

    def load(self) -> None:
        raise NotImplementedError()

    def save(self) -> None:
        if dist.get_rank() == 0:
            model_config = copy.deepcopy(self.model.speculator.config)
            model_to_save = MLPSpeculator(model_config)
            parameters_to_save = model_to_save.parameters()
        else:
            parameters_to_save = [None for param in self.model.speculator.parameters()]

        is_z3 = self.trainer.model.zero_optimization_stage() == 3

        torch.cuda.empty_cache()
        dist.barrier()

        # Gather final model.
        for parameter_to_save, (name, ds_param) in zip(
            parameters_to_save, self.model.speculator.named_parameters()
        ):
            # Using gathered parameter does not work.
            # Parameters tracking is messed up at this point
            # So we need to be selective when partitioning
            # This should oes not affect correctness.
            if is_z3 and hasattr(ds_param, "ds_id"):
                ds_param.all_gather(param_list=[ds_param])
                if dist.get_rank() == 0:
                    parameter_to_save.data.copy_(ds_param.data)
                if (
                    not ds_param.ds_active_sub_modules
                    and ds_param.ds_status is not ZeroParamStatus.INFLIGHT
                ):
                    ds_param.partition(param_list=[ds_param])
            else:
                if dist.get_rank() == 0:
                    parameter_to_save.data.copy_(ds_param.data)

        logger.info(f"Model saving at {self.config.output_dir}")
        if dist.get_rank() == 0 and self.config.output_dir is not None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            save_path = os.path.join(self.config.output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), save_path)
            model_config.save(self.config.output_dir)

        dist.barrier()
