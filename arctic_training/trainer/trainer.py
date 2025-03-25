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

import random
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import copy
import deepspeed
import numpy as np
import torch
from deepspeed.accelerator import get_accelerator
from deepspeed.utils.timer import SynchronizedWallClockTimer
from devtools import debug
from tqdm import tqdm
from transformers import set_seed
from wandb.sdk.wandb_run import Run as WandbRun

from arctic_training.callback.logging import post_loss_log_cb
from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.callback.wandb import init_wandb_project_cb
from arctic_training.callback.wandb import log_wandb_loss_cb
from arctic_training.callback.wandb import teardown_wandb_cb
from arctic_training.checkpoint.engine import CheckpointEngine
from arctic_training.config.trainer import TrainerConfig
from arctic_training.data.factory import DataFactory
from arctic_training.logging import logger
from arctic_training.model.factory import ModelFactory
from arctic_training.optimizer.factory import OptimizerFactory
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method
from arctic_training.scheduler.factory import SchedulerFactory
from arctic_training.tokenizer.factory import TokenizerFactory
from arctic_training.debug import print_rank0, print_rank, exit, debug_gathered_tensor, see_memory_usage, pr, pr0
from arctic_training.utils import get_local_rank, is_global_main_process, StepFlopCounter, gather_sum_number, format_human_base2_number, gather_object

try:
    from transformers.integrations.deepspeed import HfDeepSpeedConfig
except ImportError:
    from transformers.deepspeed import HfDeepSpeedConfig


# x = None
# def reenter():
#     global x
#     if x is None:
#         raise ValueError("shouldn't have been called 2nd time")
#     #assert x is None, "shouldn't have been called 2nd time"
#     x = 1

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

### Ulysses ###

import deepspeed.comm as dist
#from deepspeed.sequence.layer import UlyssesAttention
from einops import rearrange

from deepspeed.sequence.layer import _DimZeroAllToAll, _SeqAllToAll


"""
Some additional Ulysses docs that perhaps should go elsewhere:

If you want to try to push the seqlen higher w/o using more gpus, try to add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (but measure the performance - it could be slower)

"""
class UlyssesAttentionHF(torch.nn.Module):
    """Re-Implementation of DistributedAttention. This implementation enforces the input shape
    to be standard [sl, bs, hc, hs] form. Any deviation from this shape will raise an error.

    The primary reason for the re-implementation is to make this less error prone, and remove what seemed like
    bugs in scenarios where batch size > 1 and when using different versions of
    flash attention each of which takes different input shape. Those should be handled by
    the actual attn implementation, and not by this module.

    Dimension annotation:
        bs   = bs
        hc   = head count
        hc_l = head count local
        hs   = head_size
        sl   = seqlen
        sl_l = seqlen local
        ws   = world_size
        em    = embedding (hidden size)
        em_l  = embedding (hidden size) local

    Arguments:
        attn: normal attention implementation from transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
        local_seq_length (int): local sequence length per GPU
        global_seq_length (int): actual sequence length
        batch_size (int): batch size
        attn_head_size (int): size of each attention head
        attn_head_count (int): total number of attention heads
        kv_head_count (int): total number of kv heads
        process_group (dist.ProcessGroup): Ulysses process group
        seq_length_is_variable (bool): whether global seqlen may change between batches
    """

    def __init__(
        self,
        attn,
        local_seq_length: int,
        global_seq_length: int,
        batch_size: int,
        attn_head_count: int,
        attn_head_size: int,
        kv_head_count: int,
        process_group: dist.ProcessGroup,
        seq_length_is_variable: bool = False,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.sp_rank = dist.get_rank(process_group)

        self.local_seq_length = local_seq_length
        self.global_seq_length = global_seq_length
        self.batch_size = batch_size
        self.seq_length_is_variable = seq_length_is_variable

        self.attn_head_size = attn_head_size
        self.attn_head_count = attn_head_count
        self.global_kv_head_count = kv_head_count

        self.local_q_head_count = attn_head_count // self.world_size

        # if we have 4 kv heads and sp 8, we need to replicate kv heads 2x
        self.kv_replication_factor = self.world_size // kv_head_count
        if self.kv_replication_factor > 1:
            self.local_kv_head_count = 1
        else:
            self.local_kv_head_count = kv_head_count // self.world_size

        print_rank0(f"{self.local_q_head_count=}", skip=False)
        print_rank0(f"{self.local_kv_head_count=}", skip=False)
        print_rank0(f"{self.kv_replication_factor=}", skip=False)
        #exit()

        if self.attn_head_count % self.world_size != 0:
            raise ValueError(f"Attention head count {attn_head_count} is not divisible by SP size {self.world_size}")
        if not (self.global_kv_head_count % self.world_size == 0 or self.world_size % self.global_kv_head_count == 0):
            raise ValueError(f"KV attention head count {self.global_kv_head_count} is not divisible by SP size {self.world_size} or vice versa")

        # XXX: working on this feature MQA and some cases of GQA
        # if self.global_kv_head_count < self.world_size:
        #     raise ValueError(f"KV attention head count < sp size ({self.global_kv_head_count} < {self.world_size}) is currently not supported but it can be implemented by replicating heads")

        # XXX: add more constraints (some might have to go outside of this module or change the API to add more arguments if they are needed here, or perhaps add a special class method that validates the outside things)
        # - global_seq_length is divisible by SP? but probably has to be sorted out before this module
        # - more?

        # [sl_l bs hc hs]
        self.required_query_shape = torch.Size([local_seq_length, \
                                                batch_size, \
                                                attn_head_count, \
                                                attn_head_size])
        self.required_key_value_shape = torch.Size([local_seq_length, \
                                                batch_size, \
                                                kv_head_count, \
                                                attn_head_size])

        # [sl bs em_l]
        self.required_context_shape = torch.Size([global_seq_length, \
                                                batch_size, \
                                                attn_head_size * attn_head_count // self.world_size])

    def _combine_local_sequences(self, query, key, value) -> Tuple[Tensor, Tensor, Tensor]:

        def combine_sequence(input, head_type):
            """
                expects inputs in shape: [sl_l bs hc hs]
                returns output in shape: [sl bs hc_l hs]

                local_head_count could be different for k,v vs q if it's not an MHA situation
            """

            print_rank0('')
            print_rank0(f"combine {head_type}: before reshape:  {input.shape=}", skip=False)

            if head_type == "q":
                local_head_count = self.local_q_head_count
            else: # kv
                local_head_count = self.local_kv_head_count

                # MQA and some GQA cases:
                if self.kv_replication_factor > 1:
                    #local_head_count *= self.kv_replication_factor
                    # replicate heads to the kv_replication_factor on hc dimension [sl_l bs hc hs] - so dim=2
                    input = input.repeat_interleave(self.kv_replication_factor, dim=2)
                    print_rank0(f"combine {head_type}: after repeat interleave:  {input.shape=}", skip=False)

            # [sl_l bs hc hs] -> [sl_l bs ws hc_l hs]
            input = input.reshape([self.local_seq_length, \
                                self.batch_size, \
                                self.world_size, \
                                local_head_count, \
                                self.attn_head_size])



            print_rank0(f"combine {head_type}: after reshape:   {input.shape=}", skip=False)

            input = rearrange(input, 'sl_l bs ws hc_l hs -> ws sl_l bs hc_l hs').contiguous()
            # print_rank0(f"combine {head_type}: after rearrange: {input.shape=}", skip=False)

            output = _DimZeroAllToAll.apply(self.process_group, input)
            #output = input
            print_rank0(f"combine {head_type}: after all2all:   {output.shape=}", skip=False)

            # [ws sl_l bs hc_l hs] -> [sl bs hc_l hs]
            output = output.reshape([self.global_seq_length, *output.shape[2:]]).contiguous()
            print_rank0(f"combine {head_type}: after reshape:   {output.shape=}", skip=False)

            # [sl bs hc_l hs]
            return output


        return combine_sequence(query, head_type="q"),  combine_sequence(key, head_type="kv"),  combine_sequence(value, head_type="kv")

    def _partition_global_sequence(self, input) -> Tensor:
        """
            expects input in shape:  [sl bs em_l]
            returns output in shape: [sl_l bs em]
        """

        # print_rank0(f"partition: before reshape:  {input.shape=}")

        # [sl bs em_l] -> [ws sl_l bs em_l]
        input = input.reshape([self.world_size, \
                            self.local_seq_length, \
                            self.batch_size, \
                            self.attn_head_size * self.attn_head_count // self.world_size]).contiguous()

        # print_rank0(f"partition: after reshape:   {input.shape=}", skip=False)
        output = _DimZeroAllToAll.apply(self.process_group, input)
        #output = input
        # print_rank0(f"partition: after all2all:   {output.shape=}", skip=False)
        output = rearrange(output, 'ws sl_l bs em_l -> sl_l bs ws em_l')
        #output = rearrange(output, 'ws sl_l bs ... -> sl_l bs ws ...')
        # print_rank0(f"partition: after rearrange: {output.shape=}")

        # [sl_l bs ws em_l] -> [sl_l bs em]
        output = output.reshape([*output.shape[:2], -1]).contiguous()
        # print_rank0(f"partition: after reshape:   {output.shape=}")

        # [sl_l bs em]
        return output


    def forward(self, module: torch.nn.Module, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            attention_mask (Tensor): Attention mask
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # HF incoming shapes are:
        # [batch_size, num_heads, seqlen, head_size]
        # UlyssesAttentionHF expects:
        # [seqlen, batch_size, num_heads, head_size]
        # print_rank0(f"{query.shape=}")
        # print_rank0(f"{key.shape=}")
        # print_rank0(f"{value.shape=}")
        # print_rank0(f"{self.required_input_shape=}")
        #print(f"XXXX {query.shape=}")
        #die
        current_local_seq_length = query.shape[2]
        if self.seq_length_is_variable and current_local_seq_length != self.required_query_shape[0]:
            self.local_seq_length = current_local_seq_length
            self.global_seq_length = current_local_seq_length * self.world_size
            # update the required seqlen shapes
            self.required_query_shape = torch.Size([self.local_seq_length] + list(self.required_query_shape)[1:])
            self.required_key_value_shape = torch.Size([self.local_seq_length] + list(self.required_key_value_shape)[1:])
            self.required_context_shape = torch.Size([self.global_seq_length] + list(self.required_context_shape)[1:])

        # print_rank0(f"forward 1 {query.shape=}")
        # print_rank0(f"forward 1 {key.shape=}")
        # print_rank0(f"forward 1 {value.shape=}")

        see_memory_usage(f"enter attn forward", force=False)

        # make the blocks contiguous as early as possible to minimize fragmentation
        query = rearrange(query, 'bs hc sl hs -> sl bs hc hs') # .contiguous()
        key = rearrange(key,     'bs hc sl hs -> sl bs hc hs') # .contiguous()
        value = rearrange(value, 'bs hc sl hs -> sl bs hc hs') # .contiguous()

        # print_rank0(f"forward 2 {query.shape=}")
        # print_rank0(f"forward 2 {key.shape=}")
        # print_rank0(f"forward 2 {value.shape=}")
        # print_rank0(f"forward 2 {self.required_query_shape=}")
        # print_rank0(f"forward 2 {self.required_key_value_shape=}")

        #print_rank0(f"{attention_mask.shape=}")
        # please don't remove the white-space vertical alignment in the error message
        assert query.shape == self.required_query_shape, \
            f"[{dist.get_rank()}]: query input tensor does not match the required shape\n             {self.required_query_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}"
        assert key.shape == value.shape == self.required_key_value_shape, \
            f"[{dist.get_rank()}]: key or value input tensor does not match the required shape\n             {self.required_key_value_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}"
        # assert query.shape == key.shape == value.shape == self.required_input_shape, \
        #     f"[{dist.get_rank()}]: One of the input tensors does not match the required shape\n             {self.required_input_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}"

        see_memory_usage(f"before combine", force=False)

        # expects: [sl_l bs hc hs]
        query_layer, key_layer, value_layer = self._combine_local_sequences(query, key, value)
        # returns: [sl bs hc_l hs]

        see_memory_usage(f"after combine", force=False)

        query_layer = rearrange(query_layer, 'sl bs hc_l hs -> bs hc_l sl hs').contiguous()
        key_layer = rearrange(key_layer,     'sl bs hc_l hs -> bs hc_l sl hs').contiguous()
        value_layer = rearrange(value_layer, 'sl bs hc_l hs -> bs hc_l sl hs').contiguous()

        #query_layer = query_layer.reshape(query_layer.shape).contiguous()

        # print_rank0(f"{query_layer.shape=}")
        # print_rank0(f"{key_layer.shape=}")
        # print_rank0(f"{value_layer.shape=}")

        # if attention_mask is not None:
        #     print_rank0(f"{attention_mask.shape=}")
        #     #print_rank0(f"{attention_mask=}")

        # XXX: stick into the trainer object
        from deepspeed.utils import groups
        sp_group = groups._get_sequence_parallel_group()
        sp_world_size = groups._get_sequence_parallel_world_size()

        # debug_gathered_tensor(query_layer, sp_group, name="query_layer")
        # debug_gathered_tensor(key_layer, sp_group, name="key_layer")
        # debug_gathered_tensor(value_layer, sp_group, name="value_layer")

        print_rank0(f"HF before real attn: {query_layer.shape=}", skip=False)
        print_rank0(f"HF before real attn: {key_layer.shape=}", skip=False)
        print_rank0(f"HF before real attn: {value_layer.shape=}", skip=False)
        print_rank0(f"HF before real attn: {torch.norm(query_layer)=}")
        print_rank0(f"HF before real attn: {torch.norm(key_layer)=}")
        print_rank0(f"HF before real attn: {torch.norm(value_layer)=}")

        # pr0(f"HF before real attn: {query_layer.shape=}", skip=False)
        # pr0(f"HF before real attn: {key_layer.shape=}", skip=False)
        # pr0(f"HF before real attn: {value_layer.shape=}", skip=False)
        # pr0(f"HF before real attn: {torch.norm(query_layer)=}", skip=False)
        # pr0(f"HF before real attn: {torch.norm(key_layer)=}", skip=False)
        # pr0(f"HF before real attn: {torch.norm(value_layer)=}", skip=False)

        #exit()


        see_memory_usage(f"before core attn", force=False)

        # crucial in the case of MQA and some GQA cases we need to fix `module.num_key_value_groups`
        # see:
        # XXX: could move this somewhere to do it only once per run
        if self.kv_replication_factor > 1:
            print_rank0(f"before: {module.num_key_value_groups=}", skip=False)
            module.num_key_value_groups = query_layer.size(-3)//key_layer.size(-3)
            print_rank0(f"after: {module.num_key_value_groups=}", skip=False)

        # expects: [bs hc_l sl hs]
        context_layer, attn_weights = self.attn(module, query_layer, key_layer, value_layer, attention_mask, *args, **kwargs)
        # returns [bs sl hc_l hs]

        see_memory_usage(f"after core attn", force=False)

        # debug_gathered_tensor(context_layer, sp_group, name="context_layer")

        # print_rank0(f"HF after real attn: {context_layer.shape=}")
        # print_rank0(f"HF after real attn: {torch.norm(context_layer)=}")

        # print_rank0(f"1 {context_layer.shape=}")
        # [bs sl hc_l hs] -> [sl bs hc_l hs]'
        context_layer = rearrange(context_layer, 'bs sl ... -> sl bs ...')
        # print_rank0(f"2 {context_layer.shape=}")
        context_layer = context_layer.reshape([*context_layer.shape[:2], -1])
        # print_rank0(f"3 {context_layer.shape=}")
        # print_rank0(f"{self.required_context_shape=}")

        assert context_layer.shape == self.required_context_shape, \
                    f"The context shape {context_layer.shape} is not as expected shape {self.required_context_shape}"

        see_memory_usage(f"before partition", force=False)

        # expects: [sl bs em_l]
        output = self._partition_global_sequence(context_layer)
        # returns: [sl_l bs em]

        see_memory_usage(f"after partition", force=False)

        # print_rank0(f"1 {output.shape=}")
        output = rearrange(output, 'sl_l bs ... -> bs sl_l ...')
        # print_rank0(f"2 {output.shape=}")

        output = output.reshape([*output.shape[:2], -1])
        # print_rank0(f"3 {output.shape=}")
        # if attn_weights is not None:
        #     print_rank0(f"{attn_weights.shape=}")

        # debug_gathered_tensor(output, sp_group, name="output")

        see_memory_usage(f"exit attn forward", force=False)

        #exit()

        # expects [bs sl em]
        return output, attn_weights


class UlyssesAttentionHFNoFrag(torch.nn.Module):
    """Re-Implementation of DistributedAttention. This implementation enforces the input shape
    to be standard [sl, bs, hc, hs] form. Any deviation from this shape will raise an error.

    The primary reason for the re-implementation is to make this less error prone, and remove what seemed like
    bugs in scenarios where batch size > 1 and when using different versions of
    flash attention each of which takes different input shape. Those should be handled by
    the actual attn implementation, and not by this module.

    Dimension annotation:
        bs   = bs
        hc   = head count
        hc_l = head count local
        hs   = head_size
        sl   = seqlen
        sl_l = seqlen local
        ws   = world_size
        em    = embedding (hidden size)
        em_l  = embedding (hidden size) local

    Arguments:
        attn: normal attention implementation from transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
        local_seq_length (int): local sequence length per GPU
        global_seq_length (int): actual sequence length
        batch_size (int): batch size
        attn_head_size (int): size of each attention head
        attn_head_count (int): total number of attention heads
        kv_head_count (int): total number of kv heads
        process_group (dist.ProcessGroup): Ulysses process group
        seq_length_is_variable (bool): whether global seqlen may change between batches
    """

    def __init__(
        self,
        attn,
        local_seq_length: int,
        global_seq_length: int,
        batch_size: int,
        attn_head_count: int,
        attn_head_size: int,
        kv_head_count: int,
        device,
        process_group: dist.ProcessGroup,
        seq_length_is_variable: bool = False,
    ) -> None:
        super().__init__()
        self.attn = attn
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.sp_rank = dist.get_rank(process_group)
        self.device = device

        self.local_seq_length = local_seq_length
        self.global_seq_length = global_seq_length
        self.batch_size = batch_size
        self.seq_length_is_variable = seq_length_is_variable

        self.attn_head_size = attn_head_size
        self.attn_head_count = attn_head_count
        self.global_kv_head_count = kv_head_count

        self.local_q_head_count = attn_head_count // self.world_size


        # if we have 4 kv heads and sp 8, we need to replicate kv heads 2x
        self.kv_replication_factor = self.world_size // kv_head_count
        if self.kv_replication_factor > 1:
            self.local_kv_head_count = 1
        else:
            self.local_kv_head_count = kv_head_count // self.world_size

        # XXX: hardcoded dtype
        buffer_size_kv = batch_size*kv_head_count*local_seq_length*attn_head_size
        buffer_size_q  = batch_size*attn_head_count*local_seq_length*attn_head_size
        self.nf_k = torch.empty(buffer_size_kv, dtype=torch.bfloat16, device=device)
        self.nf_v = torch.empty(buffer_size_kv, dtype=torch.bfloat16, device=device)
        self.nf_q = torch.empty(buffer_size_q,  dtype=torch.bfloat16, device=device)

        print_rank0(f"{self.local_q_head_count=}", skip=False)
        print_rank0(f"{self.local_kv_head_count=}", skip=False)
        print_rank0(f"{self.kv_replication_factor=}", skip=False)
        #exit()

        if self.attn_head_count % self.world_size != 0:
            raise ValueError(f"Attention head count {attn_head_count} is not divisible by SP size {self.world_size}")
        if not (self.global_kv_head_count % self.world_size == 0 or self.world_size % self.global_kv_head_count == 0):
            raise ValueError(f"KV attention head count {self.global_kv_head_count} is not divisible by SP size {self.world_size} or vice versa")

        # XXX: working on this feature MQA and some cases of GQA
        # if self.global_kv_head_count < self.world_size:
        #     raise ValueError(f"KV attention head count < sp size ({self.global_kv_head_count} < {self.world_size}) is currently not supported but it can be implemented by replicating heads")

        # XXX: add more constraints (some might have to go outside of this module or change the API to add more arguments if they are needed here, or perhaps add a special class method that validates the outside things)
        # - global_seq_length is divisible by SP? but probably has to be sorted out before this module
        # - more?

        # [sl_l bs hc hs]
        self.required_query_shape = torch.Size([local_seq_length, \
                                                batch_size, \
                                                attn_head_count, \
                                                attn_head_size])
        self.required_key_value_shape = torch.Size([local_seq_length, \
                                                batch_size, \
                                                kv_head_count, \
                                                attn_head_size])

        # [sl bs em_l]
        self.required_context_shape = torch.Size([global_seq_length, \
                                                batch_size, \
                                                attn_head_size * attn_head_count // self.world_size])

    def _combine_local_sequences(self, query, key, value) -> Tuple[Tensor, Tensor, Tensor]:

        def combine_sequence(input, head_type):
            """
                expects inputs in shape: [sl_l bs hc hs]
                returns output in shape: [sl bs hc_l hs]

                local_head_count could be different for k,v vs q if it's not an MHA situation
            """

            print_rank0('')
            print_rank0(f"combine {head_type}: before reshape:  {input.shape=}", skip=False)

            if head_type == "q":
                local_head_count = self.local_q_head_count
            else: # kv
                local_head_count = self.local_kv_head_count

                # MQA and some GQA cases:
                if self.kv_replication_factor > 1:
                    #local_head_count *= self.kv_replication_factor
                    # replicate heads to the kv_replication_factor on hc dimension [sl_l bs hc hs] - so dim=2
                    input = input.repeat_interleave(self.kv_replication_factor, dim=2)
                    print_rank0(f"combine {head_type}: after repeat interleave:  {input.shape=}", skip=False)

            # [sl_l bs hc hs] -> [sl_l bs ws hc_l hs]
            input = input.reshape([self.local_seq_length, \
                                self.batch_size, \
                                self.world_size, \
                                local_head_count, \
                                self.attn_head_size])



            print_rank0(f"combine {head_type}: after reshape:   {input.shape=}", skip=False)

            input = rearrange(input, 'sl_l bs ws hc_l hs -> ws sl_l bs hc_l hs').contiguous()
            # print_rank0(f"combine {head_type}: after rearrange: {input.shape=}", skip=False)

            output = _DimZeroAllToAll.apply(self.process_group, input)
            #output = input
            print_rank0(f"combine {head_type}: after all2all:   {output.shape=}", skip=False)

            # [ws sl_l bs hc_l hs] -> [sl bs hc_l hs]
            output = output.reshape([self.global_seq_length, *output.shape[2:]]).contiguous()
            print_rank0(f"combine {head_type}: after reshape:   {output.shape=}", skip=False)

            # [sl bs hc_l hs]
            return output


        return combine_sequence(query, head_type="q"),  combine_sequence(key, head_type="kv"),  combine_sequence(value, head_type="kv")

    def _partition_global_sequence(self, input) -> Tensor:
        """
            expects input in shape:  [sl bs em_l]
            returns output in shape: [sl_l bs em]
        """

        # print_rank0(f"partition: before reshape:  {input.shape=}")

        # [sl bs em_l] -> [ws sl_l bs em_l]
        input = input.reshape([self.world_size, \
                            self.local_seq_length, \
                            self.batch_size, \
                            self.attn_head_size * self.attn_head_count // self.world_size]).contiguous()

        # print_rank0(f"partition: after reshape:   {input.shape=}", skip=False)
        output = _DimZeroAllToAll.apply(self.process_group, input)
        #output = input
        # print_rank0(f"partition: after all2all:   {output.shape=}", skip=False)
        output = rearrange(output, 'ws sl_l bs em_l -> sl_l bs ws em_l')
        #output = rearrange(output, 'ws sl_l bs ... -> sl_l bs ws ...')
        # print_rank0(f"partition: after rearrange: {output.shape=}")

        # [sl_l bs ws em_l] -> [sl_l bs em]
        output = output.reshape([*output.shape[:2], -1]).contiguous()
        # print_rank0(f"partition: after reshape:   {output.shape=}")

        # [sl_l bs em]
        return output


    def forward(self, module: torch.nn.Module, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            attention_mask (Tensor): Attention mask
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # HF incoming shapes are:
        # [batch_size, num_heads, seqlen, head_size]
        # UlyssesAttentionHF expects:
        # [seqlen, batch_size, num_heads, head_size]
        # print_rank0(f"{query.shape=}")
        # print_rank0(f"{key.shape=}")
        # print_rank0(f"{value.shape=}")
        # print_rank0(f"{self.required_input_shape=}")
        #print(f"XXXX {query.shape=}")
        #die
        current_local_seq_length = query.shape[2]
        if self.seq_length_is_variable and current_local_seq_length != self.required_query_shape[0]:
            self.local_seq_length = current_local_seq_length
            self.global_seq_length = current_local_seq_length * self.world_size
            # update the required seqlen shapes
            self.required_query_shape = torch.Size([self.local_seq_length] + list(self.required_query_shape)[1:])
            self.required_key_value_shape = torch.Size([self.local_seq_length] + list(self.required_key_value_shape)[1:])
            self.required_context_shape = torch.Size([self.global_seq_length] + list(self.required_context_shape)[1:])

        # print_rank0(f"forward 1 {query.shape=}")
        # print_rank0(f"forward 1 {key.shape=}")
        # print_rank0(f"forward 1 {value.shape=}")

        see_memory_usage(f"enter attn forward", force=False)

        # make the blocks contiguous as early as possible to minimize fragmentation
        query = rearrange(query, 'bs hc sl hs -> sl bs hc hs') # .contiguous()
        self.nf_q = self.nf_q.narrow(0, 0, query.numel())
        self.nf_q.copy_(query.view(-1))
        query.data = self.nf_q.data.view_as(query)

        key = rearrange(key,     'bs hc sl hs -> sl bs hc hs') # .contiguous()
        value = rearrange(value, 'bs hc sl hs -> sl bs hc hs') # .contiguous()

        # print_rank0(f"forward 2 {query.shape=}")
        # print_rank0(f"forward 2 {key.shape=}")
        # print_rank0(f"forward 2 {value.shape=}")
        # print_rank0(f"forward 2 {self.required_query_shape=}")
        # print_rank0(f"forward 2 {self.required_key_value_shape=}")

        #print_rank0(f"{attention_mask.shape=}")
        # please don't remove the white-space vertical alignment in the error message
        assert query.shape == self.required_query_shape, \
            f"[{dist.get_rank()}]: query input tensor does not match the required shape\n             {self.required_query_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}"
        assert key.shape == value.shape == self.required_key_value_shape, \
            f"[{dist.get_rank()}]: key or value input tensor does not match the required shape\n             {self.required_key_value_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}"
        # assert query.shape == key.shape == value.shape == self.required_input_shape, \
        #     f"[{dist.get_rank()}]: One of the input tensors does not match the required shape\n             {self.required_input_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}"

        see_memory_usage(f"before combine", force=False)

        # expects: [sl_l bs hc hs]
        query_layer, key_layer, value_layer = self._combine_local_sequences(query, key, value)
        # returns: [sl bs hc_l hs]

        see_memory_usage(f"after combine", force=False)

        query_layer = rearrange(query_layer, 'sl bs hc_l hs -> bs hc_l sl hs').contiguous()
        key_layer = rearrange(key_layer,     'sl bs hc_l hs -> bs hc_l sl hs').contiguous()
        value_layer = rearrange(value_layer, 'sl bs hc_l hs -> bs hc_l sl hs').contiguous()

        #query_layer = query_layer.reshape(query_layer.shape).contiguous()

        # print_rank0(f"{query_layer.shape=}")
        # print_rank0(f"{key_layer.shape=}")
        # print_rank0(f"{value_layer.shape=}")

        # if attention_mask is not None:
        #     print_rank0(f"{attention_mask.shape=}")
        #     #print_rank0(f"{attention_mask=}")

        # XXX: stick into the trainer object
        from deepspeed.utils import groups
        sp_group = groups._get_sequence_parallel_group()
        sp_world_size = groups._get_sequence_parallel_world_size()

        # debug_gathered_tensor(query_layer, sp_group, name="query_layer")
        # debug_gathered_tensor(key_layer, sp_group, name="key_layer")
        # debug_gathered_tensor(value_layer, sp_group, name="value_layer")

        print_rank0(f"HF before real attn: {query_layer.shape=}", skip=False)
        print_rank0(f"HF before real attn: {key_layer.shape=}", skip=False)
        print_rank0(f"HF before real attn: {value_layer.shape=}", skip=False)
        print_rank0(f"HF before real attn: {torch.norm(query_layer)=}")
        print_rank0(f"HF before real attn: {torch.norm(key_layer)=}")
        print_rank0(f"HF before real attn: {torch.norm(value_layer)=}")

        # pr0(f"HF before real attn: {query_layer.shape=}", skip=False)
        # pr0(f"HF before real attn: {key_layer.shape=}", skip=False)
        # pr0(f"HF before real attn: {value_layer.shape=}", skip=False)
        # pr0(f"HF before real attn: {torch.norm(query_layer)=}", skip=False)
        # pr0(f"HF before real attn: {torch.norm(key_layer)=}", skip=False)
        # pr0(f"HF before real attn: {torch.norm(value_layer)=}", skip=False)

        #exit()


        see_memory_usage(f"before core attn", force=False)

        # crucial in the case of MQA and some GQA cases we need to fix `module.num_key_value_groups`
        # see:
        # XXX: could move this somewhere to do it only once per run
        if self.kv_replication_factor > 1:
            print_rank0(f"before: {module.num_key_value_groups=}", skip=False)
            module.num_key_value_groups = query_layer.size(-3)//key_layer.size(-3)
            print_rank0(f"after: {module.num_key_value_groups=}", skip=False)

        # expects: [bs hc_l sl hs]
        context_layer, attn_weights = self.attn(module, query_layer, key_layer, value_layer, attention_mask, *args, **kwargs)
        # returns [bs sl hc_l hs]

        see_memory_usage(f"after core attn", force=False)

        # debug_gathered_tensor(context_layer, sp_group, name="context_layer")

        # print_rank0(f"HF after real attn: {context_layer.shape=}")
        # print_rank0(f"HF after real attn: {torch.norm(context_layer)=}")

        # print_rank0(f"1 {context_layer.shape=}")
        # [bs sl hc_l hs] -> [sl bs hc_l hs]'
        context_layer = rearrange(context_layer, 'bs sl ... -> sl bs ...')
        # print_rank0(f"2 {context_layer.shape=}")
        context_layer = context_layer.reshape([*context_layer.shape[:2], -1])
        # print_rank0(f"3 {context_layer.shape=}")
        # print_rank0(f"{self.required_context_shape=}")

        assert context_layer.shape == self.required_context_shape, \
                    f"The context shape {context_layer.shape} is not as expected shape {self.required_context_shape}"

        see_memory_usage(f"before partition", force=False)

        # expects: [sl bs em_l]
        output = self._partition_global_sequence(context_layer)
        # returns: [sl_l bs em]

        see_memory_usage(f"after partition", force=False)

        # print_rank0(f"1 {output.shape=}")
        output = rearrange(output, 'sl_l bs ... -> bs sl_l ...')
        # print_rank0(f"2 {output.shape=}")

        output = output.reshape([*output.shape[:2], -1])
        # print_rank0(f"3 {output.shape=}")
        # if attn_weights is not None:
        #     print_rank0(f"{attn_weights.shape=}")

        # debug_gathered_tensor(output, sp_group, name="output")

        see_memory_usage(f"exit attn forward", force=False)

        #exit()

        # expects [bs sl em]
        return output, attn_weights




from typing import Callable, List, Optional, Tuple, Union
from torch import nn

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

class LlamaAttentionNew(torch.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # [2, 0, -1, 128]
        # print_rank(f"{hidden_states.shape=}", ranks=[0,3], skip=False)
        # print_rank(f"{hidden_shape=}", ranks=[0,3], skip=False)
        # print_rank(f"{self.head_dim=}", ranks=[0,3], skip=False)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # print_rank(f"{query_states.shape=}", skip=False)
        # print_rank(f"{key_states.shape=}", skip=False)
        # print_rank(f"{value_states.shape=}", skip=False)
        #query_states = query_states.contiguous()
        cos, sin = position_embeddings
        # print_rank(f"{cos.shape=}", skip=False)
        # print_rank(f"{sin.shape=}", skip=False)
        # cos1 = cos.unsqueeze(1).contiguous()
        # print_rank(f"{cos1.shape=}", skip=False)

        # q_embed = (query_states * cos1) # + (rotate_half(query_states) * sin)


        #exit()

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        import transformers
        #attention_interface: Callable = transformers.models.llama.modeling_llama.eager_attention_forward

        # XXX: fix me - temp hardcoding - must remove this hack
        #self.config._attn_implementation = "ulysses"

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        #print(ALL_ATTENTION_FUNCTIONS.keys())
        #print(f"{self.config._attn_implementation=}")
        #exit()

        # XXX: meanwhile for consistency with non-sp testing
        #attention_interface: Callable = transformers.models.llama.modeling_llama.eager_attention_forward
        #attention_interface: Callable = transformers.integrations.flash_attention.flash_attention_forward
        #attention_interface: Callable = transformers.integrations.sdpa_attention.sdpa_attention_forward

        # XXX:
        # if "ulysses" in ALL_ATTENTION_FUNCTIONS:
        #     attention_interface = ALL_ATTENTION_FUNCTIONS["ulysses"]
        #     print_rank0(f"custom attention on {torch.distributed.get_rank()}")

        print_rank0(f"HF before attn: {query_states.shape=}")
        print_rank0(f"HF before attn: {key_states.shape=}")
        print_rank0(f"HF before attn: {value_states.shape=}")
        print_rank0(f"HF before attn: {torch.norm(query_states)=}")
        print_rank0(f"HF before attn: {torch.norm(key_states)=}")
        print_rank0(f"HF before attn: {torch.norm(value_states)=}")

        # pr0(f"HF before attn: {query_states.shape=}")
        # pr0(f"HF before attn: {key_states.shape=}")
        # pr0(f"HF before attn: {value_states.shape=}")
        # pr0(f"HF before attn: {torch.norm(query_states)=}")
        # pr0(f"HF before attn: {torch.norm(key_states)=}")
        # pr0(f"HF before attn: {torch.norm(value_states)=}")
        # exit()

        if attention_mask is not None:
            print_rank0(f"HF before attn: {attention_mask.shape=}")
        else:
            print_rank0(f"HF before attn: {attention_mask=}")

        #print_rank0(f"HF before attn: {value_states=}")

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        print_rank0(f"HF after attn: {attn_output.shape=}")
        print_rank0(f"HF after attn: {torch.norm(attn_output)=}")
        # pr0(f"HF after attn: {attn_output.shape=}")
        # pr0(f"HF after attn: {torch.norm(attn_output)=}")
        if attn_weights is not None:
            print_rank0(f"HF after attn: {attn_weights.shape=}")
            print_rank0(f"HF after attn: {torch.norm(attn_weights)=}")

        #exit()

        from deepspeed.utils import groups
        sp_group = groups._get_sequence_parallel_group()
        sp_world_size = groups._get_sequence_parallel_world_size()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        if sp_world_size > 1:
            debug_gathered_tensor(attn_output, sp_group, name="attn_output prefinal")
        else:
            print_rank0(f"HF after attn prefinal: {attn_output.shape=}")
            print_rank0(f"HF after attn prefinal: {torch.norm(attn_output)=}")

        attn_output = self.o_proj(attn_output)

        if sp_world_size > 1:
            debug_gathered_tensor(attn_output, sp_group, name="attn_output after o_proj")
        else:
            print_rank0(f"HF after o_proj: {attn_output.shape=}")
            print_rank0(f"HF after o_proj: {torch.norm(attn_output)=}")
        #exit()

        return attn_output, attn_weights

class Trainer(ABC, CallbackMixin, metaclass=RegistryMeta):
    """Base Trainer class."""

    name: str
    """
    Name of the trainer used for registering custom trainers. This name
    should be unique and is used in the training recipe YAMLs to identify which
    trainer to be used.
    """

    config: TrainerConfig
    """
    The type of the config class that the trainer uses. This should be a
    subclass of TrainerConfig and add any trainer-specific fields.
    """

    data_factory: DataFactory
    """
    A List of valid data factory types that the trainer can use. These should
    inherit from DataFactory. The first item in the list will be used as the
    default if the type is not explicitly set in the YAML config.
    """

    model_factory: ModelFactory
    """
    A List of valid model factory types that the trainer can use. These should
    inherit from ModelFactory. The first item in the list will be used as the
    default if the type is not explicitly set in the YAML config.
    """

    checkpoint_engine: CheckpointEngine
    """
    A List of valid checkpoint engine types that the trainer can use. These
    should inherit from CheckpointEngine. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    optimizer_factory: OptimizerFactory
    """
    A List of valid optimizer factory types that the trainer can use. These
    should inherit from OptimizerFactory. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    scheduler_factory: SchedulerFactory
    """
    A List of valid scheduler factory types that the trainer can use. These
    should inherit from SchedulerFactory. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    tokenizer_factory: TokenizerFactory
    """
    A List of valid tokenizer factory types that the trainer can use. These
    should inherit from TokenizerFactory. The first item in the list will be
    used as the default if the type is not explicitly set in the YAML config.
    """

    callbacks: List[Tuple[str, Callable]] = [
        post_loss_log_cb,
        init_wandb_project_cb,
        log_wandb_loss_cb,
        teardown_wandb_cb,
    ]
    """
    A list of callbacks for the trainer. Callbacks are specified as tuples of a
    string indicating where the callback should be placed and a callable that
    implements the callback. Callback events for the trainer include `pre-` and
    `post-` for `init`, `train`, `epoch`, `step`, and `checkpoint`.
    """

    @classmethod
    def _validate_subclass(cls) -> None:
        _validate_class_attribute_set(cls, "name")
        _validate_class_attribute_type(cls, "config", TrainerConfig)
        _validate_class_attribute_type(cls, "data_factory", DataFactory)
        _validate_class_attribute_type(cls, "model_factory", ModelFactory)
        _validate_class_attribute_type(cls, "checkpoint_engine", CheckpointEngine)
        _validate_class_attribute_type(cls, "optimizer_factory", OptimizerFactory)
        _validate_class_attribute_type(cls, "scheduler_factory", SchedulerFactory)
        _validate_class_attribute_type(cls, "tokenizer_factory", TokenizerFactory)
        _validate_class_method(cls, "loss", ["self", "batch"])
        _validate_class_method(cls, "step", ["self", "batch"])
        _validate_class_method(cls, "epoch", ["self"])
        _validate_class_method(cls, "train", ["self"])
        _validate_class_method(cls, "checkpoint", ["self"])

    def __init__(self, config: TrainerConfig) -> None:

        logger.info(f"Initializing Trainer with config:\n{debug.format(config)}")
        self.config = config
        self.epoch_idx = 0
        self.train_batch_idx = 0
        self.global_step = 0
        self.eval_batch_idx = 0
        self.early_stop = False
        self.world_size = config.world_size
        self.global_rank = config.global_rank
        self.training_finished = False
        self.wandb_experiment: Optional[WandbRun] = None

        self._set_seeds(self.config.seed)

        tokenizer_factory = self.config.tokenizer.factory(self)
        self.tokenizer = tokenizer_factory()

        data_factory = self.config.data.factory(self)
        self.train_dataloader, self.eval_dataloader = data_factory()

        # from arctic_training.utils import get_local_rank
        # self.local_rank = get_local_rank()
        # if self.local_rank == 0:
        #     data_factory = self.config.data.factory(self)
        # dist.barrier()
        # if self.local_rank != 0:
        #     data_factory = self.config.data.factory(self)

        # XXX: eventually switch back to normal hf modeling code (it's just debug prints mod'ed at the moment)
        # there are no functional code changes in LlamaAttentionNew
        import transformers.models.llama.modeling_llama
        #transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttentionNew

        # XXX: find a place for this code
        if self.config.sequence_parallel_size == 1:
            mpu = None
        else:
            import arctic_training.trainer.parallel_state as mpu
            import torch
            from transformers import AutoConfig
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            #from transformers.modeling_utils import AttentionInterface

            print_rank0(f"MPU INIT on rank {torch.distributed.get_rank()}")
            print_rank0(f"MBS  {self.config.micro_batch_size}")
            mpu.initialize_model_parallel(sequence_parallel_size=self.config.sequence_parallel_size)

            # we don't have the model yet at this stage
            hf_model_config = AutoConfig.from_pretrained(self.config.model.name_or_path)

            core_attn_implementation = self.config.model.attn_implementation
            if core_attn_implementation == "eager":
                # The problem is that `eager` wants an attention_mask and it creates the wrong attention mask it seems if we don't provide one - it's possible that we could somehow solve this, but it's also unlikely someone will want to use the slow eager with sequence parallelism

                # XXX: there is also flex attention but I haven't tested if it works
                raise ValueError(f"{core_attn_implementation} attn_implementation isn't currently supported by sequence parallelism. Set attention_implementation to either 'flash_attention_2' and 'sdpa'.")

            core_attn_function = ALL_ATTENTION_FUNCTIONS.get(core_attn_implementation, None)
            if core_attn_function is None:
                raise ValueError(f"{core_attn_implementation} is not a valid attn_implementation. The choices are {ALL_ATTENTION_FUNCTIONS.keys()}")

            #attn_implementation_real =  transformers.models.llama.modeling_llama.eager_attention_forward
            #attn_implementation_real = transformers.integrations.flash_attention.flash_attention_forward
            #attn_implementation_real = transformers.integrations.sdpa_attention.sdpa_attention_forward

            # print_rank(core_attn_implementation)
            # print_rank(core_attn_function)
            # exit()

            #from deepspeed.sequence.layer import DistributedAttention
#            uattn = UlyssesAttentionHFNoFrag(
            uattn = UlyssesAttentionHF(
                attn=core_attn_function,
                local_seq_length=self.config.data.max_length // mpu.get_sequence_parallel_world_size(),
                global_seq_length=self.config.data.max_length,
                batch_size=self.config.micro_batch_size,
                attn_head_count=hf_model_config.num_attention_heads,
                attn_head_size=hf_model_config.hidden_size // hf_model_config.num_attention_heads,
                kv_head_count=hf_model_config.num_key_value_heads,
                #device=self.device,
                process_group=mpu.get_sequence_parallel_group(),
                seq_length_is_variable=True,
            )

            def uattn_wrapper(
                module: torch.nn.Module,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: torch.Tensor,
                *args,
                **kwargs,
            ) -> Tuple[torch.Tensor, torch.Tensor]:

                # XXX: we are relaying on position_ids for SP to work so attention_mask has to be None
                # the problem is that HF currently doesn't know anything about ALL_ATTENTION_FUNCTIONS["ulysses"] so it doesn't make a special case like for "flash_attention_2" and "sdpa" and it creates an attention mask on the fly and it breaks things.
                attention_mask = None

                attn_output, attn_weights = uattn(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    # XXX: fixme
                    *args,
                    **kwargs
                )
                return attn_output, attn_weights

            # we are overriding the original core attn implementation with `ulysses` and we have already passed the original core attn implementation to `UlyssesAttentionHF`
            self.config.model.attn_implementation = "ulysses"
            ALL_ATTENTION_FUNCTIONS["ulysses"] = uattn_wrapper

        #rint(self.config.model.attn_implementation)
        #exit()

        dschf = HfDeepSpeedConfig(self.config.deepspeed)  # noqa: F841
        model_factory = self.config.model.factory(self)
        self.model = model_factory()

        # sanity check - w/o a proper HF API it's too easy to lose the attn_implementation override
        if self.config.sequence_parallel_size > 1:
            if self.model.config._attn_implementation != "ulysses":
                raise ValueError("sequence parallelism has been configured but the HF model isn't configured to run it - check the injection")

        optimizer_factory = self.config.optimizer.factory(self)
        self.optimizer = optimizer_factory()

        scheduler_factory = self.config.scheduler.factory(self)
        self.scheduler = scheduler_factory()

        self.step_timer = SynchronizedWallClockTimer.Timer("step")
        self.iter_timer = SynchronizedWallClockTimer.Timer("iteration")

        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            args=self.config,
            lr_scheduler=self.scheduler,
            config=self.config.deepspeed,
            mpu=mpu,
        )

        # import deepspeed
        # import torch
        # from transformers import AutoModel
        # with deepspeed.utils.init_on_device.OnDevice(dtype=torch.bfloat16, device='meta'):
        #     meta_model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        #     self.meta_model, *_ = deepspeed.initialize(
        #     model=meta_model,
        #     optimizer=self.optimizer,
        #     args=self.config,
        #     lr_scheduler=self.scheduler,
        #     config=self.config.deepspeed,
        #     mpu=mpu,
        # )


        self.checkpoint_engines = [
            engine(self) for engine in self.config.checkpoint_engines
        ]

        for engine in self.checkpoint_engines:
            if engine.config.auto_resume:
                engine.load(self.model)

    def _set_seeds(self, seed: int) -> None:
        logger.info(f"Setting random seeds to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        set_seed(seed)

    @property
    def epochs(self) -> tqdm:
        """Epochs iterator."""
        return tqdm(
            range(self.epoch_idx, self.config.epochs),
            desc="Epochs",
            unit="epoch",
            disable=self.global_rank != 0,
        )

    @property
    def train_batches(self) -> tqdm:
        """Training data iterator."""
        return tqdm(
            self.train_dataloader,
            desc="Train Batches",
            unit="batch",
            disable=self.global_rank != 0,
        )

    @property
    def device(self) -> torch.device:
        """Current device."""
        return torch.device(get_accelerator().device_name(self.global_rank))

    @property
    def model_unwrapped(self):
        """Return the original model before it was wrapped by deepspeed"""

        # XXX: later might add a recursion if we have more than one level of wrapping
        if hasattr(self.model, "module"):
            return self.model.module
        else:
            return self.model

    @property
    def training_horizon(self) -> int:
        """Total number of training iterations."""
        if self.train_dataloader is None:
            raise ValueError("Train dataloader not initialized.")
        if self.config.train_iters:
            return self.config.train_iters
        return (
            self.config.epochs
            * len(self.train_dataloader)
            // self.config.gradient_accumulation_steps
        )

    @property
    def warmup_steps(self) -> int:
        """Number of warmup steps."""
        return int(self.config.scheduler.warmup_ratio * self.training_horizon)

    @callback_wrapper("loss")
    @abstractmethod
    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Loss function for the trainer. This method should be implemented by the
        inheriting trainer class.
        """
        raise NotImplementedError("Loss method must be implemented by the trainer.")

    @callback_wrapper("step")
    def step(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Step function for the trainer. Each batch of training data is passed to
        this method.
        """
        self.step_timer.start()

        #import deepspeed.comm as dist
        # import q
        #from deepspeed.utils import groups
        # q(self.global_rank)
        # print_rank0(f"{groups._get_sequence_parallel_group()=}")
        # print_rank0(f"{groups._get_sequence_parallel_rank()=}")
        # print_rank0(f"{groups._get_sequence_parallel_world_size()=}")
        #dist.barrier()
        #import time
        #time.sleep(5)
        #die

        torch.set_printoptions(sci_mode=False)
        # torch.set_printoptions(
        #     threshold=100000000, # print all data (without ... skipping) - can be huge!
        #     sci_mode=False,      # print all data on the same scale of 1 (this disables scientific notation)
        #     precision=6,         # print X decimal points for floats (default 4)
        #     edgeitems=5,         # when the data is large and skipped, control how many entries are printed on each edge
        #     linewidth=120,       # redefine linewidth for when lines are \n-wrapped in printout (default 80)
        #                         # if threshold is defined, matrix printing will ignore this setting
        #     profile="full",      # printing defaults: "default", "short", "full"
        # )

        # if self.global_rank == 0:
        #     print_rank0(batch)

        see_memory_usage("before forward", force=False)

        self.model.train()
        if self.config.sequence_parallel_size == 1:
        #     avg_loss = self.model.backward(loss)

            loss = self.loss(batch)
            #self.model.backward(loss)


        else:
            # sp will do backward inside loss
            loss = self.loss(batch)
        see_memory_usage("after backward", force=False)

        self.model.step()

        # import math
        # if not math.isnan(loss):
        #     print(f"{loss=}")
        #     self.model.step()

        see_memory_usage("after step", force=False)
        #exit()

            # # should loss be averaged over sp sub-steps and logged as such?
            # loss = loss_aggregate / sp_world_size
            # print_rank0(f"averaged loss = {loss}")
            # #exit()


        # if 1:
        #     # XXX: probably need to do padding so that all sequence chunks are the same?!
        #     import math
        #     print_rank0(f"{len(batch['input_ids'][0])=}")
        #     #print_rank0(f"{len(batch['input_ids'][1])=}")
        #     #seq_length = len(batch["input_ids"][0])
        #     seq_length = self.config.data.max_length

        #     sp_world_size = groups._get_sequence_parallel_world_size()
        #     sp_rank = groups._get_sequence_parallel_rank()
        #     chunk_len = math.ceil(seq_length / sp_world_size)
        #     print_rank0(f"{seq_length=}")
        #     print_rank0(f"{chunk_len=}")

        #     # this is the original chunking logic
        #     for k in batch.keys():
        #         if sp_world_size > 1 and k in ["input_ids", "position_ids"]: # , "labels"]:
        #         #if sp_world_size > 1 and k in ["input_ids"]:
        #             batch[k] = batch[k][:, chunk_len*sp_rank:chunk_len*(sp_rank+1)].to(self.device)
        #         else:
        #             batch[k] = batch[k].to(self.device)
        #         print_rank0(f"{k} {batch[k].shape=}")


        # else:
        #     # non-sp original version
        #     self.model.train()
        #     # XXX: fixme
        #     #self.global_step = self.model.global_steps
        #     loss = self.loss(batch)
        #     print_rank(f"{self.train_batch_idx}: {loss.requires_grad=}")
        #     print_rank(f"{self.train_batch_idx}: {loss=}")

        #     #self.model.backward(loss)
        #     avg_loss = self.model.backward(loss)
        #     print_rank0(f"zero loss: {avg_loss}")

        from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param
        # print_rank0(f"!!! {torch.norm(safe_get_full_fp32_param(self.model.lm_head.weight))} lm_head.weight", skip=False)
        # print_rank0(f"!!! {torch.norm(safe_get_full_fp32_param(self.model.model.layers[0].self_attn.q_proj.weight))} q.weight", skip=False)

        # # print_rank(f"end loss = {loss}")
        # print_rank0(f"!!! {torch.norm(safe_get_full_grad(self.model.module.lm_head.weight))} lm_head.grad", skip=False)
        # print_rank0(f"!!! {torch.norm(safe_get_full_grad(self.model.module.model.layers[0].self_attn.q_proj.weight))} q.grad", skip=False)
        #exit()

        # for n, p in self.model.named_parameters():
        #     print_rank(f"!!! {torch.norm(safe_get_full_fp32_param(p)):6.2f} {n}", skip=False)
        # for n, p in self.model.named_parameters():
        #     print_rank(f"!!! {torch.norm(safe_get_full_grad(p)):6.2f} {n}", skip=False)
        # for n, p in self.model.named_parameters():
        #     nans = torch.isnan(p).sum()
        #     infs = torch.isinf(p).sum()
        #     if nans > 0: print_rank(f"!!! Got NANs {nans} {n}", skip=False)
        #     if infs > 0: print_rank(f"!!! Got INFs {infs} {n}", skip=False)

        #print(f"ITERATION {loss=}")

        # use deepspeed global step as golden truth
        self.global_step = self.model.global_steps
        if self.global_step >= self.training_horizon:
            self.early_stop = True

        self.checkpoint()

        if (
            self.config.exit_iteration > 0
            and self.config.exit_iteration == self.global_step
        ):
            self.early_stop = True
            logger.info(f"Hit exit iteration of {self.global_step}, ending training")

        self.step_timer.stop()
        step_time_secs = self.step_timer.elapsed() / 1000
        if self.config.step_timer:
            logger.info(f"step time: {step_time_secs} secs")

        return loss, step_time_secs

    @callback_wrapper("epoch")
    def epoch(self) -> None:
        """
        Epoch training loop. This method will be called for each epoch of
        training and iterates across batches of training data, calling the step
        method on each batch.
        """

        self.iter_timer.start()

        # enable memory history, which will
        # add tracebacks and event history to snapshots
        mem_profiler = False
        if mem_profiler:
            MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000
            torch.cuda.memory._record_memory_history()#max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)

        # XXX: this counter must not be reset between epochs
        self.train_batch_idx = 0
        for batch in self.train_batches:
            self.train_batch_idx += 1
            print_rank(f"\n\n\n\n\nITERATION: {self.train_batch_idx} ", skip=False)

            #continue

            # if (batch["position_ids"] == 0).sum() > 1:
            #     print("{self.train_batch_idx} run into a packed sample, skipping")
            #     continue

            # if self.train_batch_idx < 8:
            #     continue
            # if self.train_batch_idx == 4:
            #     exit()
            # if self.train_batch_idx == 8:
            #     continue

            #print_rank(f"{self.tokenizer.decode(batch['input_ids'][0])=}", skip=False, force=True)
            #exit()

            # print_rank(f"before gather: : {batch['labels'].shape=}", skip=False)
            # print_rank(f"before gather: : {batch['position_ids'].shape=}", skip=False)
                #print_rank0(f"before gather: {k}: {batch[k]=}")
            #exit()

            # since different ranks may have different batches of different seqlen for the correct flop counting we need the total seqlen across all ranks
            #
            # since we could have sp>1+dp>1 gather across all ranks
            seqlen_local = len(batch["input_ids"][0])
            gathered_seqlen_total = gather_sum_number(seqlen_local, device=self.device, group=None)
            gathered_seqlens = gather_object(seqlen_local, device=self.device, group=None)

            #see_memory_usage("before step", force=True)

            # # XXX: need to gather seqlen from all ranks as it's not guaranteed to be the same
            # orig_model = self.model
            # self.model = self.meta_model
            # with self.step_flos_counter(self.train_batch_idx, cache_key=seqlen_local):
            #     loss, step_time_secs = self.step(batch)
            # self.model = orig_model

            # # XXX: need to gather seqlen from all ranks as it's not guaranteed to be the same
            # with self.step_flos_counter(self.train_batch_idx, cache_key=seqlen_local):
            #     loss, step_time_secs = self.step(batch)
            loss, step_time_secs = self.step(batch)

            #see_memory_usage("after step", force=True)

            from deepspeed.utils import groups
            sp_group = groups._get_sequence_parallel_group()
            sp_world_size = groups._get_sequence_parallel_world_size()

            # per gpu
            # tflos = self.step_flos_counter.get_total_tflos()
            # #tflos *= sp_world_size # bug in FlopCounterMode

            # print(f"measured {tflos=}")
            # #print(f"{step_time_secs=}")

            #gathered_step_tflos = gather_sum_number(tflos, device=self.device)
            #gathered_step_tflos = gather_sum_number(0, device=self.device)

            # XXX: this is sort of pointless, since all gpus are synced - so a local measurement is already the same elsewhere usually
            gathered_step_time_total = gather_sum_number(step_time_secs, device=self.device)
            gathered_step_times = gather_object(step_time_secs, device=self.device)
            #print(gathered_step_times)

            # gathered_step_tflops = gathered_step_tflos / gathered_step_time_total * sp_world_size
            gathered_step_time_mean = gathered_step_time_total / self.world_size

            import functools

            from functools import wraps
            # https://stackoverflow.com/a/78988160/9201239
            def memoize_2nd_arg(func):
                """Memoize like functools.cache, but only cache based on the 2nd argument.
                this version only works with *args calling interface - will not work with **kwargs
                """
                cache = func.cache = {}

                @wraps(func)
                def memoizer(arg1, arg2, *args):
                    if arg2 not in cache:
                        cache[arg2] = func(arg1, arg2, *args)
                    return cache[arg2]
                return memoizer

            #@memoize_2nd_arg
            def estimate_tflos(model, seq_len):
                """
                this is an estimator for a collective computation across SP ranks (divide the result by SP size to get for one rank) with the result adapted to a single gpu

                it assumes dtype bf16 (2 bytes) and recalculation of activations (could make it a parameter - will be a multiplier of 3 instead of 4 then. 4 is 2 fwd + 2 bwd, 3 is 1 fwd + 2 bwd)

                Formulae: (seq * model_size * 2 * 4 + num_layers * seq * seq * hidden_size * 2 * 2 * 4) / 1e12

                model: unwrapped model
                seq_len: 1 batch sample seqlen

                it expects a deepspeed zero sharded model
                returns estimated tflos computed by one gpu
                """
                def numel_fn(p):
                    return p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
                model_size = sum(numel_fn(p) for p in model.parameters())
                num_layers = model.config.num_hidden_layers
                hidden_size = model.config.hidden_size
                # print(f"{model_size=}")
                # print(f"{num_layers=}")
                # print(f"{hidden_size=}")
                # print(f"{seq_len=}")
                tflos_estimated = (seq_len * model_size * 2 * 4 + num_layers * seq_len * seq_len * hidden_size * 2 * 2 * 4 ) / 1e12

                return tflos_estimated

            # because of the seq**2 in the formula calculate flos for each seqlen separately and sum up
            tflos_estimated_total = sum(estimate_tflos(self.model_unwrapped, gathered_seqlens[rank]) for rank in range(self.world_size))
            # XXX: what happens when it's dp=2 sp=4? which world size are we caclulating over?
            tflos = tflos_estimated_total / self.world_size
            # print(f"estimated {tflos=}")
            # print(f"{tflos_estimated_total=}")
            # print(f"{gathered_step_time_total=}")

            #dist.barrier()

            # try to get the iteration timer stop as close as possible to the point of end of logging
            # alternatively could put it at the end of the logger and let the next iteration absorb the previous iteration's logging overhead
            self.iter_timer.stop()
            iter_time_secs = self.iter_timer.elapsed() / 1000
             # any of the remaining logging overhead will get counted in the next iteration along with the DL.iter().next()
            self.iter_timer.start()

            iter_time_total = iter_time_secs * self.world_size
            step_tflops = tflos_estimated_total / gathered_step_time_total
            iter_tflops = tflos_estimated_total / iter_time_total

            # XXX: this should become the point where we actually log train data in one go
            # but the problem is that we are still iterating over epochs so there could be all kinds of reset side-effects - watch this
            train_log_data = dict(
                iter=self.model.global_steps,
                iter_tflops=iter_tflops,
                iter_time=iter_time_secs,
                step_tflops=step_tflops,
                step_time=gathered_step_time_mean,
                loss=loss,
                lr=self.model.lr_scheduler.get_last_lr()[0],
                seqlen_total=gathered_seqlen_total,
            )

            # the reason for a special list is 2-fold:
            # 1. to allow logging only some metrics and not all in the dense one line log
            # 2. to put the key metrics first
            # but the full log can be dumped to include all logging data
            train_log_key_order = [
                "iter",
                "loss",
                "iter_time",
                "step_time",
                "iter_tflops",
                "step_tflops",
                "lr",
                "seqlen_total",
            ]
            # can skip entries that require no special formatting
            # XXX: perhaps the values can be a list like [fmt, str]? that way we can automatically add the measurement identifier, [".1f", "secs"] and [".2f", "TFLOPS"]? resulting in "0.50 secs" and "450 TFLOPS"? on the other hand the name of the field often implies what it is - not sure
            train_log_key_fmt = dict(
                iter_tflops=".1f",
                step_tflops=".1f",
                iter="",
                loss=".4f",
                lr=".4E",
                step_time=".4f",
                iter_time=".4f",
            )

            # we log to wandb unconditionally every step
            if self.wandb_experiment is not None:
                self.wandb_experiment.log(train_log_data, step=self.model.global_steps)

            # XXX: add `train_log_interval` to config and integrate it here, hardcoding for now
            # once added this code will need to change to accumulate, average, etc - e.g. say if we interval=10
            train_log_interval = 1
            if (is_global_main_process() and self.train_batch_idx % train_log_interval == 0
                and self.train_batch_idx != 1 # don't log step 0 as it is a massive outlier and messes up plots like time
            ):


                # This is Megatron-LM style dense human redable essentials logging - one iteration per iterval
                # how can we automate this? a special format time where it's a fn instead of a string format
                train_log_data["seqlen_total"] = format_human_base2_number(train_log_data["seqlen_total"], suffix="")
                log_line = ""
                # XXX: currently missing progress indication 5/1000 (1%)
                for k in train_log_key_order:
                    fmt = train_log_key_fmt.get(k, "")
                    log_line += f"{k}: {train_log_data[k]:{fmt}} | "
                print(log_line)

            if self.early_stop:
                break

        if mem_profiler:
            torch.cuda.memory._dump_snapshot(f"mem/mem_snapshot.{self.global_rank}.pickle")

    @callback_wrapper("train")
    def train(self) -> None:
        """
        Main training loop. Calls the epoch method for each epoch of training.
        """

        #self.step_flos_counter = StepFlopCounter(start_iter=2)

        try:
            for epoch_idx in self.epochs:
                self.epoch_idx = epoch_idx
                self.epoch()
                if self.early_stop:
                    break
                self.checkpoint()
            self.training_finished = True
            logger.info("Training finished.")
            self.checkpoint()
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            # logger.info(f"{self._trainer_state}")
            raise (e)

    @callback_wrapper("checkpoint")
    def checkpoint(self) -> None:
        for engine in self.checkpoint_engines:
            if engine.do_checkpoint:
                logger.info(f"Saving Checkpoint at global step: {self.global_step}.")
                engine.save(self.model)
