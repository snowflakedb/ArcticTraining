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

from typing import Union

import torch

from arctic_training.checkpoint.ds_engine import DSCheckpointEngine
from arctic_training.checkpoint.hf_engine import HFCheckpointEngine
from arctic_training.data.sft_factory import SFTDataFactory
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.model.liger_factory import LigerModelFactory
from arctic_training.optimizer.adam_factory import CPUAdamOptimizerFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.scheduler.hf_factory import HFSchedulerFactory
from arctic_training.tokenizer.hf_factory import HFTokenizerFactory
from arctic_training.trainer.trainer import Trainer
from arctic_training.trainer.utils import to_device
from arctic_training.debug import see_memory_usage, print_rank0, print_rank


class SFTTrainer(Trainer):
    name = "sft"
    data_factory: SFTDataFactory
    model_factory: Union[HFModelFactory, LigerModelFactory]
    checkpoint_engine: Union[DSCheckpointEngine, HFCheckpointEngine]
    optimizer_factory: Union[FusedAdamOptimizerFactory, CPUAdamOptimizerFactory]
    scheduler_factory: Union[HFSchedulerFactory]
    tokenizer_factory: Union[HFTokenizerFactory]

    def loss(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)
        outputs = self.model(**batch, use_cache=False)
        loss = outputs.loss
        return loss

    # XXX: return tensor like normal `loss`?
    def sp_fwd_loss_bwd(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)

        # XXX: this will be later moved to get instantiated right after deepspeed.initialize with a new Trainer method `self.post_deepspeed_initialize` or something like that. It'd auto-manifest `sp_fwd_loss_bwd` method which will do what's here now but the `ulysses` object will be instantiated once per training
        from arctic_training.trainer.trainer import UlyssesAttentionHFFwdLossBwdWithLogits
        ulysses = UlyssesAttentionHFFwdLossBwdWithLogits(
            model=self.model,
            model_unwrapped=self.model_unwrapped,
            device=self.device,
            num_loss_logit_shards="auto",
        )
        return ulysses.sp_fwd_loss_bwd(batch)
