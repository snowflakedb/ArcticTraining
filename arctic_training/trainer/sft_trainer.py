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

from typing import Dict
from typing import Union

import torch

from arctic_training.checkpoint.ds_engine import DSCheckpointEngine
from arctic_training.checkpoint.hf_engine import HFCheckpointEngine
from arctic_training.data.sft_factory import SFTDataFactory
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.model.liger_factory import LigerModelFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.registry import register
from arctic_training.scheduler.hf_factory import HFSchedulerFactory
from arctic_training.tokenizer.hf_factory import HFTokenizerFactory
from arctic_training.trainer import Trainer
from arctic_training.debug import print_rank0, print_rank, exit

def to_device(batch: Dict, device: str) -> Dict:
    output = {}
    for k, v in batch.items():
        output[k] = v.to(device)
    return output


@register
class SFTTrainer(Trainer):
    name = "sft"
    data_factory: SFTDataFactory
    model_factory: Union[HFModelFactory, LigerModelFactory]
    checkpoint_engine: Union[DSCheckpointEngine, HFCheckpointEngine]
    optimizer_factory: Union[FusedAdamOptimizerFactory]
    scheduler_factory: Union[HFSchedulerFactory]
    tokenizer_factory: Union[HFTokenizerFactory]

    def loss_sp(self, batch) -> torch.Tensor:

        #from torch.nn import CrossEntropyLoss
        #loss_function = CrossEntropyLoss()

        from deepspeed.utils import groups
        import torch.distributed as dist
        #from transformers.modeling_utils import unwrap_model
        #unwrapped_model = unwrap_model(self.model)
        batch = to_device(batch, self.device)

        # remove labels to avoid an attempt to calculate loss by transformers
        labels = batch.pop("labels")
        outputs = self.model(**batch, use_cache=False)

        logits = outputs.logits
        print_rank(f"{torch.norm(logits)=}")
        print_rank(f"{logits.shape=}")
        #print_rank(f"{logits.dtype=}")
        print_rank(f"{labels.shape=}")

        # XXX: stick into the trainer object
        sp_group = groups._get_sequence_parallel_group()
        sp_world_size = groups._get_sequence_parallel_world_size()

        tensor_list = [torch.zeros_like(logits) for _ in range(sp_world_size)]
        dist.all_gather(tensor_list, logits, group=sp_group)
        for t in tensor_list:
            print_rank(f"{t.shape=}")

        # concatenate on the seqlen dimension        
        logits = torch.cat(tensor_list, dim=1)
        print_rank(f"after cat: {logits.shape=}")

        # XXX: is this more efficient?
        # logits_shape = [logits.shape[0], logits.shape[1]*sp_world_size, logits.shape[2]]
        # tensor_out = torch.zeros(logits_shape, dtype=logits.dtype, device=logits.device)
        # dist.all_gather_into_tensor(tensor_out, labels, group=sp_group)

        # print(outputs)
        # output = self.tokenizer.batch_decode(outputs)
        # #if self.global_rank == 0:
        # print(f"RANK {self.global_rank}: {output}")

        #print(self.model.module.config.keys())
        #import sys; sys.exit()

        loss = self.model.loss_function(logits=logits, labels=labels, vocab_size=self.model_unwrapped.config.vocab_size)
        print_rank(f"{loss=}")

        return loss

    def loss(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)
        outputs = self.model(**batch, use_cache=False)

        # print(outputs)
        # output = self.tokenizer.batch_decode(outputs)
        # #if self.global_rank == 0:
        # print(f"RANK {self.global_rank}: {output}")

        loss = outputs.loss
        return loss
