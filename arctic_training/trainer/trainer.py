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
from functools import cached_property
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import copy
import deepspeed
import numpy as np
import torch
import wandb
from deepspeed.accelerator import get_accelerator
from devtools import debug
from tqdm import tqdm
from transformers import set_seed
from wandb.sdk.wandb_run import Run as WandbRun
import torch.distributed.nn

from arctic_training.callback.logging import post_loss_log_cb
from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.checkpoint.engine import CheckpointEngine
from arctic_training.config.trainer import TrainerConfig
from arctic_training.data.factory import DataFactory
from arctic_training.logging import logger
from arctic_training.metrics import Metrics
from arctic_training.model.factory import ModelFactory
from arctic_training.optimizer.factory import OptimizerFactory
from arctic_training.registry import RegistryMeta
from arctic_training.registry import _validate_class_attribute_set
from arctic_training.registry import _validate_class_attribute_type
from arctic_training.registry import _validate_class_method
from arctic_training.scheduler.factory import SchedulerFactory
from arctic_training.tokenizer.factory import TokenizerFactory
from arctic_training.debug import print_rank0, print_rank, exit, debug_gathered_tensor, see_memory_usage, pr, pr0
from arctic_training.utils import StepFlopCounter
from arctic_training.config.utils import get_local_rank

from transformers.integrations.deepspeed import HfDeepSpeedConfig

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

### Ulysses ###

import deepspeed.comm as dist
#from deepspeed.sequence.layer import UlyssesAttention
from einops import rearrange


from deepspeed.sequence.layer import _SeqAllToAll
## XXX: when creating a PR into deepspeed move _DimZeroAllToAll into deepspeed.sequence.layer
#from deepspeed.sequence.layer import _DimZeroAllToAll, _SeqAllToAll
'''Differentiable All2All across dimension 0.'''
class _DimZeroAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:
        world_size = dist.get_world_size(group)
        assert input.shape[0] == world_size, f"Dim 0 {input.shape[0]} is not world size"

        ctx.group = group

        output = torch.empty_like(input).contiguous()
        dist.all_to_all_single(output, input.contiguous(), group=group)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _DimZeroAllToAll.apply(ctx.group, *grad_output))

"""
Some additional Ulysses docs that perhaps should go elsewhere:

If you want to try to push the seqlen higher w/o using more gpus, try to add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (but measure the performance - it could be slower). This should help with minimizing fragmentation.

"""
class UlyssesSPAttentionHF(torch.nn.Module):
    """Re-Implementation of DistributedAttention. This implementation enforces the input shape
    to be standard [sl, bs, hc, hs] form. Any deviation from this shape will raise an error.

    The primary reason for the re-implementation is to make this less error prone, and remove what seemed like
    bugs in scenarios where batch size > 1 and when using different versions of
    flash attention each of which takes different input shape. Those should be handled by
    the actual attn implementation, and not by this module.

    This class then has been further adapted to work with HF Transformers' supported attention mechanism.

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
        num_hidden_layers (int): total number of layers
        process_group (dist.ProcessGroup): Ulysses process group
        seq_length_is_variable (bool): whether global seqlen may change between batches


    Extras:
        - set self.skip_all_but_last_attention_debug_mode to True to enable fast debug which will skip calling all core attn layers but the last one, it will produce garbage of course quality-wise.
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
        num_hidden_layers: int,
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

        self.num_hidden_layers = num_hidden_layers
        self.skip_all_but_last_attention_debug_mode = False
        self.rotating_layer_counter = 0 # used for dev work

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
            #see_memory_usage(f"combine: 1", force=False)
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
            #see_memory_usage(f"combine: 2", force=False)

            # [sl_l bs hc hs] -> [sl_l bs ws hc_l hs]
            input = input.reshape([self.local_seq_length, \
                                self.batch_size, \
                                self.world_size, \
                                local_head_count, \
                                self.attn_head_size])

            #see_memory_usage(f"combine: 3", force=False)

            print_rank0(f"combine {head_type}: after reshape:   {input.shape=}", skip=False)

            input = rearrange(input, 'sl_l bs ws hc_l hs -> ws sl_l bs hc_l hs').contiguous()
            # print_rank0(f"combine {head_type}: after rearrange: {input.shape=}", skip=False)
            #see_memory_usage(f"combine: 4", force=False)

            output = _DimZeroAllToAll.apply(self.process_group, input)
            #output = input
            print_rank0(f"combine {head_type}: after all2all:   {output.shape=}", skip=False)
            #see_memory_usage(f"combine: 5", force=False)

            # [ws sl_l bs hc_l hs] -> [sl bs hc_l hs]
            output = output.reshape([self.global_seq_length, *output.shape[2:]]).contiguous()
            print_rank0(f"combine {head_type}: after reshape:   {output.shape=}", skip=False)
            #see_memory_usage(f"combine: 6", force=False)

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
        # UlyssesSPAttentionHF expects:
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

        if not self.skip_all_but_last_attention_debug_mode:
            # expects: [bs hc_l sl hs]
            context_layer, attn_weights = self.attn(module, query_layer, key_layer, value_layer, attention_mask, *args, **kwargs)
            # returns [bs sl hc_l hs]
        else:
            # we need this hack during development in order to be able to check memory fitting w/o waiting for 3h to compute 1.5M seqlen attention, because it's quadratic in dense attention, so we skip all but the last core attention call - we want the last one to still get the memory usage approximately close to the real memory usage.
            # of course the loss will be wrong when we do that.
            self.rotating_layer_counter = (self.rotating_layer_counter + 1) % self.num_hidden_layers
            # we detect the last layer by module counting since we know how many layers there are
            if self.rotating_layer_counter % self.num_hidden_layers == 0:
                #print(f"{self.rotating_layer_counter} Real")
                # do the real pass
                context_layer, attn_weights = self.attn(module, query_layer, key_layer, value_layer, attention_mask, *args, **kwargs)
            else:
                #print(f"{self.rotating_layer_counter} Fake")
                # this feeds bogus data of the right shape - good enough for quick debug
                context_layer = rearrange(query_layer, 'bs hc_l sl ... -> bs sl hc_l ...')
                attn_weights = None

        # print(f"{context_layer.shape=}")
        # if attn_weights is not None:
        #     print(f"{attn_weights.shape=}")
        # else:
        #     print(f"attn_weights=None")

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


    @classmethod
    def register_with_transformers(cls, model_name_or_path, core_attn_implementation, sequence_parallel_size, max_length, micro_batch_size, seq_length_is_variable=True):
        """
        Register "ulysses" attn_implementation with HF transformers and return mpu (Megatron-LM-style parallel state object).
        If sequence_parallel_size==1 do nothng and return None.

        """
        if sequence_parallel_size == 1:
            return None

        #see_memory_usage("ulysses: 1", force=True)
        import arctic_training.trainer.parallel_state as mpu
        #import torch
        from transformers import AutoConfig
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        #see_memory_usage("ulysses: 1.1", force=True)
        # print_rank0(f"MPU INIT on rank {torch.distributed.get_rank()}")
        # print_rank0(f"MBS  {micro_batch_size}")
        mpu.initialize_model_parallel(sequence_parallel_size=sequence_parallel_size)
        #see_memory_usage("ulysses: 1.2", force=True)
        # we don't have the model yet at this stage
        hf_model_config = AutoConfig.from_pretrained(model_name_or_path)
        #see_memory_usage("ulysses: 1.3", force=True)
        if core_attn_implementation not in ['flash_attention_2', 'sdpa']:
            # - eager: The problem is that `eager` wants an attention_mask and it creates the wrong attention mask it seems if we don't provide one - it's possible that we could somehow solve this, but it's also unlikely someone will want to use the slow eager attention with sequence parallelism
            # - flex_attention: haven't tried
            # - flash_attention_2: with some models leads to loss=nan when using packed samples - works fine w/o packed samples

            raise ValueError(f"{core_attn_implementation} attn_implementation isn't currently supported by Ulysses sequence parallelism. Set attn_implementation to either 'flash_attention_2' and 'sdpa'.")

        if core_attn_implementation not in ALL_ATTENTION_FUNCTIONS:
            raise ValueError(f"{core_attn_implementation} is not a valid attn_implementation. The choices are {ALL_ATTENTION_FUNCTIONS.valid_keys()}")
        core_attn_function = ALL_ATTENTION_FUNCTIONS[core_attn_implementation]
        #see_memory_usage("ulysses: 3", force=True)
        uattn = UlyssesSPAttentionHF(
            attn=core_attn_function,
            local_seq_length=max_length // mpu.get_sequence_parallel_world_size(),
            global_seq_length=max_length,
            batch_size=micro_batch_size,
            attn_head_count=hf_model_config.num_attention_heads,
            attn_head_size=hf_model_config.hidden_size // hf_model_config.num_attention_heads,
            kv_head_count=hf_model_config.num_key_value_heads,
            num_hidden_layers=hf_model_config.num_hidden_layers,
            #device=self.device,
            process_group=mpu.get_sequence_parallel_group(),
            seq_length_is_variable=seq_length_is_variable,
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

        ALL_ATTENTION_FUNCTIONS.register("ulysses", uattn_wrapper)
        #see_memory_usage("ulysses: 4", force=True)
        #exit()
        return mpu

    @classmethod
    def validate_model(cls, model, sequence_parallel_size):
        if sequence_parallel_size > 1:
            if model.config._attn_implementation != "ulysses":
                raise ValueError("sequence parallelism has been configured but the HF model isn't configured to run it - check whether the `register_with_transformers` method was called before the `model` has been created")



from collections import defaultdict
from torch.utils.data import DataLoader
class UlyssesSPDataLoaderWrapper():
    def __init__(
        self,
        dl: DataLoader,
        sp_rank: int,
        sp_group,
        sp_world_size,
        device,
    ):
        """
        Assumption: the batch is a dict with at least the keys: `input_ids`, `labels`, `position_ids` - but can have any additional keys necessary.
        """

        self.dl = dl
        self.sp_rank = sp_rank
        self.sp_group = sp_group
        self.sp_world_size = sp_world_size
        self.device = device

        self.iter = iter(dl)
        self.micro_batches = []

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.micro_batches) == 0:
            self.refill()

        batch = self.micro_batches.pop(0)

        seq_length = len(batch['input_ids'][0])

        if seq_length % self.sp_world_size != 0:
            raise ValueError(f"batch's seqlen={seq_length} isn't divisible by sp-size={self.sp_world_size}")
        chunk_len = int(seq_length / self.sp_world_size)

        # because we have to gather logits from all sp ranks we have to do the loss function ourselves
        # therefore remove labels to avoid an attempt to calculate loss by transformers
        labels = batch.pop("labels")
        labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
        batch["shift_labels"] = labels[..., 1:].contiguous()
        # free up temp memory
        del labels

        # batch sharding
        for k in batch.keys():
            #print_rank(f"SLICING {k} {chunk_len=}: {self.sp_rank=}", skip=False)
            batch[k] = batch[k][:, chunk_len*self.sp_rank:chunk_len*(self.sp_rank+1)].to(self.device)
            # else:
            #     print_rank(f"KEEPING {k} {batch[k].shape=}", skip=False)
            #     batch[k] = batch[k].to(self.device)

            #print_rank0(f"after sp: {k}: {batch[k].shape=}")

        # if len(self.micro_batches) == 0:
        #     raise StopIteration
        return batch

    def refill(self):
        # this will raise StopIteration when empty
        batch = next(self.iter)
        micro_batches = defaultdict(dict)
        # XXX: replace with more efficient all-to-all?

        # we have batches of variable seqlen so in order to do all_gather on batches - we need to know the exact length of each tensor on each rank
        seqlen = torch.tensor(batch["input_ids"].shape[1], dtype=torch.int64, device=self.device)
        #print(seqlen)
        seqlens = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(self.sp_world_size)]
        dist.all_gather(seqlens, seqlen, group=self.sp_group)
        seqlens = [x[0].item() for x in seqlens]

        for k in batch.keys():
            batch[k] = batch[k].to(self.device)
            #print_rank(f"before gather: {k}: {batch[k].shape=}", skip=False)
            #print_rank0(f"before gather: {k}: {batch[k]=}")
            with torch.no_grad():

                tensor_list = [torch.zeros((batch[k].shape[0],seqlens[i]), dtype=batch[k].dtype, device=batch[k].device) for i in range(self.sp_world_size)]
                # # print(tensor_list)
                # # print(batch[k])
                dist.all_gather(tensor_list, batch[k], group=self.sp_group)

                # gathering on the data dimension
                # will be concatenating and later splitting again for the more general case
                # batch[k] = torch.cat(tensor_list, dim=1)
                for rank, tensor in enumerate(tensor_list):
                    micro_batches[rank][k] = tensor
                    #print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k].shape=}", skip=False)
                    #print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k].device=}", skip=False)
                    #print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k]=}", skip=False)
                    # if k == "input_ids":
                    #     print_rank0(f"{self.trainer.tokenizer.decode(micro_batches[rank][k][0])=}", skip=False)

                #see_memory_usage("mid-gathering", force=False)

        del tensor_list
        del batch

        # convert to list
        self.micro_batches = [micro_batches[i] for i in range(len(micro_batches))]


# XXX: this class shouldn't depend on anything in AT (trainer, etc) - we can have a subclass if needed to support that
# but can also accept kwargs that a customizer can use in methods

#import torch
import math
import deepspeed.comm as dist
from collections import defaultdict
class UlyssesSPFwdLossBwdWithLogits():
    def __init__(self,
                 model,
                 model_unwrapped,
                 device,
                 num_loss_logit_shards="auto",
                 **kwargs
        ):

        self.model = model
        self.model_unwrapped = model_unwrapped
        self.device = device
        self.num_loss_logit_shards = num_loss_logit_shards
        self.kwargs = kwargs

        from deepspeed.utils import groups
        self.sp_group = groups._get_sequence_parallel_group()
        self.sp_world_size = groups._get_sequence_parallel_world_size()
        self.sp_rank = groups._get_sequence_parallel_rank()


    def sp_fwd_loss_bwd(self, batch) -> torch.Tensor:

        see_memory_usage(f"entered sp_fwd_loss_bwd", force=True)

        # ensure shapes are correct
        if not (batch["input_ids"].shape == batch["position_ids"].shape == batch["labels"].shape):
            raise ValueError(f'Borked batch {batch["input_ids"].shape=} != {batch["position_ids"].shape=} != {batch["labels"].shape=}) in DataLoader->iter->next, cannot continue with Ulysses Sequence parallelism')

        # gather DL batches into super-batches
        # Important: DL doesn't always yield max_length batches. Different ranks may have different seqlen and each could be <= max_length (but always divisible by 256)

        micro_batches = defaultdict(dict)
        # Efficient gathering of batch inputs across ranks:
        # The problem is that our DL doesn't guarantee the same seqlen on all ranks and may give, 3x 1024 and 1x 768 on 4 gpus for max_length 1024. so 3 options we have to be able to gather batches are:
        # 1. use all_gather_object - which allows different shapes - but potentially introducing an undesired overhead - 2x pickle calls
        # 2. use all_gather and change DL pad to make sure that all ranks always get the same input shape - this creates its own overhead since if we say have ranks with seqlen 512, 768, 1024, 1024 - now we will need to process 4x 1024 seqlens
        # 3. use all_gather and post gathering truncate tensors to their intended length - another overhead of allocating and truncating tensors
        # using approach (1) for now but might want to benchmark later the other 2 approaches

        see_memory_usage("before gathering", force=False)
        #print_rank(f"{self.trainer.tokenizer.decode(batch['input_ids'][0])=}", skip=False)
        #exit()

        #dist.barrier(group=self.sp_group)
        #see_memory_usage("after barrier", force=False)

        # XXX: if using all_gather_object we can gather the whole batch at once and not per-key! so can drop the loop for that approach

        # we have batches of variable seqlen so in order to do all_gather on batches - we need to know the exact length of each tensor on each rank
        seqlen = torch.tensor(batch["input_ids"].shape[1], dtype=torch.int64, device=self.device)
        #print(seqlen)
        seqlens = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in range(self.sp_world_size)]
        dist.all_gather(seqlens, seqlen, group=self.sp_group)
        seqlens = [x[0].item() for x in seqlens]
        #print(seqlens)
        #exit()

        for k in batch.keys():
            batch[k] = batch[k].to(self.device)
            print_rank(f"before gather: {k}: {batch[k].shape=}", skip=False)
            #print_rank0(f"before gather: {k}: {batch[k]=}")
            with torch.no_grad():
                # tensor_list = [torch.zeros_like(batch[k]) for _ in range(self.sp_world_size)]
                # dist.all_gather(tensor_list, batch[k], group=self.sp_group)

                tensor_list = [torch.zeros((batch[k].shape[0],seqlens[i]), dtype=batch[k].dtype, device=batch[k].device) for i in range(self.sp_world_size)]
                # # print(tensor_list)
                # # print(batch[k])
                dist.all_gather(tensor_list, batch[k], group=self.sp_group)

                # tensor_list = [None for _ in range(self.sp_world_size)]
                # torch.distributed.all_gather_object(tensor_list, batch[k], group=self.sp_group)

                # gathering on the data dimension
                # will be concatenating and later splitting again for the more general case
                # batch[k] = torch.cat(tensor_list, dim=1)
                for rank, tensor in enumerate(tensor_list):
                    micro_batches[rank][k] = tensor
                    print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k].shape=}", skip=False)
                    #print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k].device=}", skip=False)
                    #print_rank(f"after gather: {rank} {k}: {micro_batches[rank][k]=}", skip=False)
                    # if k == "input_ids":
                    #     print_rank0(f"{self.trainer.tokenizer.decode(micro_batches[rank][k][0])=}", skip=False)

                #see_memory_usage("mid-gathering", force=False)

        del tensor_list
        del batch


        #exit()
        # loss_aggregate = 0
        # we need to chunk twice - each time on SP size level
        # - the first time is because we artifically made the seqlen SP-times longer
        # - the second time is because of the Ulysses algorithm

        see_memory_usage("after gathering", force=False)


        self.model.set_gradient_accumulation_boundary(False)

        losses = []
        for sub_step_id in range(self.sp_world_size):
            #print(f"{sub_step_id=}")
            # if sub_step_id == 1:
            #     continue
            # if sub_step_id == 3:
            #     break


            batch = micro_batches[sub_step_id]

            see_memory_usage(f"{sub_step_id=} start", force=False)
            #print_rank0(batch)

            import math
            print_rank0(f"{sub_step_id}: {len(batch['input_ids'][0])=}")
            seq_length = len(batch['input_ids'][0])
            #seq_length = self.config.data.max_length

            if seq_length % self.sp_world_size != 0:
                raise ValueError(f"{sub_step_id=}: batch's seqlen={seq_length} isn't divisible by sp-size={self.sp_world_size}")
            ##chunk_len = math.ceil(seq_length / self.sp_world_size)
            chunk_len = int(seq_length / self.sp_world_size)
            print_rank0(f"{sub_step_id=}: {seq_length=}")
            print_rank0(f"{sub_step_id=}: {chunk_len=}")

            # to enable the correct mean calculation across shards before sharding the micro batch:
            # 1. count the number of non- `-100`` elements per shard
            # 2. and subtract one more element because of label shifting
            non_skipped_items = {}
            for rank in range(self.sp_world_size):
                non_skipped = (batch["labels"][:, chunk_len*rank:chunk_len*(rank+1)] != -100).sum().item()
                if non_skipped > 1:
                    non_skipped -= 1
                non_skipped_items[rank] = non_skipped
            print_rank(f"{non_skipped_items=}", skip=False)


            # because we have to gather logits from all sp ranks we have to do the loss function ourselves
            # therefore remove labels to avoid an attempt to calculate loss by transformers
            labels = batch.pop("labels")
            labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
            batch["shift_labels"] = labels[..., 1:].contiguous()
            # free up temp memory
            del labels

            # batch sharding
            for k in batch.keys():
                print_rank(f"SLICING {k} {chunk_len=}: {self.sp_rank=}", skip=False)
                batch[k] = batch[k][:, chunk_len*self.sp_rank:chunk_len*(self.sp_rank+1)].to(self.device)
                # else:
                #     print_rank(f"KEEPING {k} {batch[k].shape=}", skip=False)
                #     batch[k] = batch[k].to(self.device)

                print_rank0(f"after sp: {k}: {batch[k].shape=}")
                #print_rank0(f"after sp: {k}: {batch[k]=}")
            #outputs = self.model(**batch, use_cache=False)
            #loss = outputs.loss
            see_memory_usage(f"{sub_step_id=} after chunking", force=False)

            # XXX: this would be the same not just for SFT so probably should abstract it away
            #from deepspeed.utils import groups
            #import torch.distributed as dist
            #import torch

            see_memory_usage(f"{sub_step_id=} before forward", force=True)

            #print_rank(f"SLICE DECODE: {sub_step_id=} {self.trainer.tokenizer.decode(batch['input_ids'][0])=}", skip=False)
            #print_rank(f"SLICE DECODE: {sub_step_id=} {batch['position_ids'][0]=}", skip=False)

            shift_labels = batch.pop("shift_labels")
            #print_rank(f"{shift_labels=}", skip=False)
            see_memory_usage(f"{sub_step_id=} after shift labels", force=False)

            outputs = self.forward(batch)
            #outputs = self.model(**batch, use_cache=False)
            logits = outputs.logits

            see_memory_usage(f"{sub_step_id=} after forward", force=False)

            #print_rank(f"{labels=}", skip=False)
            #print_rank(f"{logits=}", skip=False)
            # print_rank(f"logit nans: {torch.isnan(logits).sum()}", skip=False)
            # print_rank(f"logit infs: {torch.isinf(logits).sum()}", skip=False)
            #see_memory_usage(f"{sub_step_id=} before loss", force=True)
            loss = self.compute_loss(labels=None, shift_labels=shift_labels)

            # if all((shift_labels == -100).squeeze()):
            #     # this is the case where all labels in the micro-batch are -100 (very common for SFT) - CE returns `nan` in this case, so we don't want to call loss and instead create a differentiable loss `0` which will also set all the grads to `0` in `backward` - the effect of this is akin to a perfect score where the model needs no adjustment since grads will be all zeros.
            #     # XXX: should this be float and not the original dtype?
            #     loss = (logits.sum() * 0.0).float()
            #     #loss = FakeLoss.apply(logits)
            # else:
            #     #import gc; gc.collect()
            #     #torch.cuda.empty_cache()
            #     #see_memory_usage(f"{sub_step_id=} before loss", force=True)
            #     #loss = self.model_unwrapped.loss_function(logits=logits, labels=None, vocab_size=self.model_unwrapped.config.vocab_size, shift_labels=shift_labels)


            #     shards = 8
            #     loss = UlyssesSPChunkedMemEfficientLoss.apply(self.model_unwrapped.loss_function, logits, self.model_unwrapped.config.vocab_size, shift_labels, shards)


                #see_memory_usage(f"{sub_step_id=} after loss", force=True)

            #loss = outputs.loss
            print_rank(f"LOSS local {loss=}", skip=False)

            # free up temp mem (e.g. outputs.logits are huge)
            del outputs

            see_memory_usage(f"{sub_step_id=} after loss", force=False)
            #exit()

            # if torch.isnan(loss):
            #     break
            #     #continue
            #     #loss = torch.tensor(0.0).to(self.device).requires_grad_() + 0.0
            # differentiable loss aggregation across ranks
            #import torch.distributed.nn.functional
            #loss = torch.distributed.nn.functional.all_reduce(loss, op=torch.distributed.ReduceOp.SUM, group=self.sp_group)
            losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=self.sp_group)
            #print(f"LOSS {losses_per_rank=}")
            print_rank(f"LOSS {losses_per_rank=}", skip=False)

            # since each shard may have a variable number of skipped elemented - need to calculate a weighted mean depending on each rank's contribution - this will also take care of loss=0 when all elements are -100 in a shard
            # XXX: not expecting a total of 0-non-skipped items for div
            loss = sum(losses_per_rank[rank] * non_skipped_items[rank] for rank in range(self.sp_world_size)) / sum(non_skipped_items.values())
            # this is a much simpler version w/o weighting
            # skip 0.0 entries when calculating total loss per batch
            # loss = torch.stack(list(l for l in losses_per_rank if l != 0)).mean()

            #loss = torch.cat([l.unsqueeze() for l in losses_per_rank], dim=0).mean()
            #loss = sum(loss_per_rank) # / self.sp_world_size
            #loss = sum(tensor_list)
            #print_rank(f"LOSS averaged {loss=}", skip=False)
            #print("LOSS", loss)
            see_memory_usage(f"{sub_step_id=} after gathered loss", force=False)

            #exit()

            #logits = outputs.logits
            #print_rank(f"{sub_step_id=}: {torch.norm(logits)=}", skip=False)
            #print_rank(f"{sub_step_id=}: {logits.shape=}")
            #print_rank(f"{logits.dtype=}")
            #print_rank(f"{sub_step_id=}: {labels.shape=}")

            # # XXX: stick into the trainer object
            # #self.sp_group = groups._get_sequence_parallel_group()
            # #self.sp_world_size = groups._get_sequence_parallel_world_size()
            # # we need the differentiable all_gather, which is the functional version of it
            # import torch.distributed.nn.functional
            # tensor_list = torch.distributed.nn.functional.all_gather(logits, self.sp_group)
            # # concatenate on the seqlen dimension
            # logits = torch.cat(tensor_list, dim=1)
            # del tensor_list
            # print_rank(f"after cat: {logits.shape=}")
            # see_memory_usage(f"{sub_step_id=} after cat", force=False)

            #print_rank(f"LOSS {logits.shape=}: {labels.shape=}", skip=False)

            # loss = self.model_unwrapped.loss_function(logits=logits, labels=labels, vocab_size=self.model_unwrapped.config.vocab_size)
            # #print_rank0(f"intermediary {loss.item()*self.sp_world_size=}")

            # # optimize memory
            # del logits
            # del labels

            # #loss = self.loss(batch)
            # loss_aggregate += loss.item()*self.sp_world_size

            #print_rank(f"{self.train_batch_idx}-{sub_step_id}: {loss.requires_grad=}")
            #print_rank(f"{self.train_batch_idx}-{sub_step_id}: {loss=}")

            see_memory_usage(f"{sub_step_id=} before backward", force=False)
            #import gc; gc.collect()
            #self.model.backward(loss)
            self.backward()

            # print_rank(f"{labels[0][70:80]=}", skip=False)
            # print_rank(f"{logits[0][70:80]=}", skip=False)
            # print_rank(f'{batch["input_ids"][0][70:80]=}', skip=False)
            # print_rank(f'{batch["input_ids"].grad[0][70:80]=}', skip=False)
            # print_rank(f"{logits.grad[0][70:80]=}", skip=False)
            # exit()

            print_rank0(f"zero loss: {loss}", skip=False)
            # print_rank0(f"zero loss: {avg_loss}", skip=False)
            see_memory_usage(f"{sub_step_id=} after backward", force=False)

            losses.append(loss.detach().item())


            # from deepspeed.utils import safe_get_full_grad
            # print_rank(f"{torch.norm(safe_get_full_grad(self.model.module.lm_head.weight))=}")
            # print_rank(f"{torch.norm(safe_get_full_grad(self.model.module.model.layers[0].self_attn.q_proj.weight))=}")

            # #w = self.model.module.model.layers[0].self_attn.q_proj.weight
            # w = self.model.module.lm_head.weight
            #from deepspeed.utils import safe_get_full_grad
            #print_rank0(f"{torch.norm(safe_get_full_grad(self.model.module.lm_head.weight))=}")
            #print_rank0(f"{torch.norm(safe_get_full_grad(self.model.module.model.layers[0].self_attn.q_proj.weight))=}")

        self.model.set_gradient_accumulation_boundary(True)

        # for per-iteration reporting
        if len(losses) == 0:
            loss = float('nan')
        else:
            loss = sum(losses) / len(losses)

        #exit()
        # XXX: temp to measure the real memory usage
        # gc_empty_cuda_cache()

        return loss

    # @classmethod
    # def next_power_of_2(cls, x):
    #     """
    #     take any number and find the next power of 2
    #     7.5 => 8
    #     8 => 8
    #     9 => 16
    #     """
    #     return 1<<(math.ceil(x)-1).bit_length()

    def forward(self, batch):
        # critical: the labels shouldn't be in batch
        outputs = self.model(**batch, use_cache=False)
        self.logits = outputs.logits
        #self.outputs = outputs
        return outputs

    def compute_loss(self, labels, shift_labels):
        if all((shift_labels == -100).squeeze()):
            # this is the case where all labels in a micro-batch are -100 (very common for SFT) - CE returns `nan` in this case, so we don't want to call loss and instead create a differentiable loss `0` which will also set all the grads to `0` in `backward` - the effect of this is akin to a perfect score where the model needs no adjustment since grads will be all zeros.
            # XXX: should this be float and not the original dtype?
            loss = (self.logits.sum() * 0.0).float()
        else:
            if self.num_loss_logit_shards == "auto":
                # parameterize to about 1GB fp32 logits shards
                slice_size_in_gb = 1 # XXX: make configurable?
                size_in_gb = self.logits.numel() * 4 / 2**30 # fp32
                # the sp shard's seqlen sp shard can be easily not divisible by the derived number of chunked loss shards, so we use the uppper ceiling and allow the last chunk to be shorter than the rest
                self.num_loss_logit_shards = math.ceil(size_in_gb / slice_size_in_gb)
                #print(f"derived {self.num_loss_logit_shards} shards for size {size_in_gb}GB")
            if self.num_loss_logit_shards > 1:
                loss = UlyssesSPChunkedMemEfficientLoss.apply(self.model_unwrapped.loss_function, self.logits, self.model_unwrapped.config.vocab_size, shift_labels, self.num_loss_logit_shards)
            else:
                # XXX: for some reason this fails with zero1
                loss = self.model_unwrapped.loss_function(logits=self.logits, labels=None, vocab_size=self.model_unwrapped.config.vocab_size, shift_labels=shift_labels)

        self.loss = loss
        return loss

    def backward(self):
        self.model.backward(self.loss)



class UlyssesSPChunkedMemEfficientLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss_fn, logits, vocab_size, shift_labels, shards) -> torch.Tensor:
        """
            logits doesn't have to be divisible by shards, the last shard will be shorter than the rest.
        """
        ctx.save_for_backward(logits, shift_labels)
        ctx.loss_fn = loss_fn
        ctx.vocab_size = vocab_size
        ctx.shards = shards

        with torch.no_grad():
            seqlen = shift_labels.shape[1]
            shard_step = math.ceil(seqlen / shards)
            loss_shards = []
            total_good_items = 0

            # since -100s are ignored we have to perform a weighted average on each loss slice as each slice may contribute a different number of non- -100 labels
            # if seqlen / shards != 0 - the last chunk is just shorter than the rest but no data is ignored
            for i in range(shards):
                # XXX: here and everywhere don't make a copy, pass the slice or perhaps narrow/view?
                shift_labels_shard = shift_labels[:,i*shard_step:(i+1)*shard_step]
                if all((shift_labels_shard == -100).squeeze()):
                    continue # ignore this shard
                loss_shard = loss_fn(
                    logits=logits[:,i*shard_step:(i+1)*shard_step,:],
                    labels=None,
                    vocab_size=vocab_size,
                    shift_labels=shift_labels_shard)
                good_items = sum((shift_labels_shard != -100).squeeze())
                loss_shards.append(loss_shard*good_items)
                total_good_items += good_items
            total_loss = torch.cat([l.unsqueeze(0) for l in loss_shards], dim=0).sum()
            weighted_loss = total_loss / total_good_items

        #weighted_loss.requires_grad = True
        return weighted_loss

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:

        logits, shift_labels = ctx.saved_tensors
        loss_fn = ctx.loss_fn
        vocab_size = ctx.vocab_size
        shards = ctx.shards

        grad = grads[0]
        logits_grad = torch.zeros_like(logits)
        #logits_grad = torch.zeros(logits.shape, device=logits.device, dtype=grad.dtype, requires_grad=logits.requires_grad)

        logits_shards       = list(torch.chunk(logits, chunks=shards, dim=1))
        shift_labels_shards = list(torch.chunk(shift_labels, chunks=shards, dim=1))
        del logits
        del shift_labels
        ctx.logits = None
        ctx.shift_labels = None
        ctx.loss_fn = None
        ctx.vocab_size = None
        ctx.shards = None

        for i in range(shards):
            logits_shard       = logits_shards.pop(0)
            shift_labels_shard = shift_labels_shards.pop(0)

            shard_offset = i * logits_shard.numel()
            # this will enable gradual population of the pre-allocated `logits_shard.grad` during `torch.autograd.backward` calls
            logits_shard.grad = logits_grad.view(-1).narrow(0, shard_offset, logits_shard.numel()).view_as(logits_shard)

            with torch.enable_grad():
                if all((shift_labels_shard == -100).squeeze()):
                    # fake loss calculation, since CE will return nan, but grads will be set
                    # a normal loss_fn upcasts logits to float so match it
                    loss_shard = (logits_shard.sum() * 0.0).float()
                else:
                    loss_shard = loss_fn(
                        logits=logits_shard.requires_grad_(),
                        labels=None,
                        vocab_size=vocab_size,
                        shift_labels=shift_labels_shard,
                    )

            torch.autograd.backward(loss_shard, grad)

        logits_grad /= shards

        #print(f"returning {logits_grad.norm()=}")
        #print(f"returning {logits_grad=}")
        # only logits (2nd arg) needs grads
        return None, logits_grad, None, None, None


class UlyssesSPAttentionHFNoFrag(torch.nn.Module):
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
        # UlyssesSPAttentionHF expects:
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
    ]
    """
    A list of callbacks for the trainer. Callbacks are specified as tuples of a
    string indicating where the callback should be placed and a callable that
    implements the callback. Callback events for the trainer include `pre-` and
    `post-` for `init`, `train`, `epoch`, `step`, and `checkpoint`.
    """

    # XXX: hack to compare correctness until we support GAS
    temp_losses = []

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
        self.epoch_finished = False
        self.training_finished = False
        self.wandb_experiment: Optional[WandbRun] = None

        self._set_seeds(self.config.seed)

        # enable memory history, which will add tracebacks and event history to snapshots
        # "none" | "e2e" | "step"
        self.mem_profiler = "none"
        #self.mem_profiler = "step"
        # profiling from here is slower, best to start at top of `epoch` ("step")
        if self.mem_profiler == "e2e":
            torch.cuda.memory._record_memory_history(max_entries=100_000)
        #see_memory_usage("before model creation", force=True)

        tokenizer_factory = self.config.tokenizer.factory(self)
        self.tokenizer = tokenizer_factory()

        #see_memory_usage("after tokenizer", force=True)

        dist.barrier()
        #see_memory_usage("before dataloader", force=True)

        data_factory = self.config.data.factory(self)
        self.train_dataloader, self.eval_dataloader = data_factory()

        #see_memory_usage("after dataloader", force=True)
        #exit()
        # XXX: eventually switch back to normal hf modeling code (it's just debug prints mod'ed at the moment)
        # there are no functional code changes in LlamaAttentionNew
        import transformers.models.llama.modeling_llama
        #transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttentionNew

        # XXX: We can abstract this section further with AT-specific wrapper, but UlyssesSPAttentionHF should not have any AT-specific objects / assumptions
        mpu = UlyssesSPAttentionHF.register_with_transformers(
            model_name_or_path=self.config.model.name_or_path,
            core_attn_implementation=self.config.model.attn_implementation,
            sequence_parallel_size=self.config.sequence_parallel_size,
            max_length=self.config.data.max_length,
            micro_batch_size=self.config.micro_batch_size,
            seq_length_is_variable=True,
        )
        if self.config.sequence_parallel_size > 1:
            # we are overriding the original core attn implementation with `ulysses` and we have already passed the original core attn implementation to `UlyssesSPAttentionHF`
            self.config.model.attn_implementation = "ulysses"

        #see_memory_usage("after ulysses", force=True)

        dschf = HfDeepSpeedConfig(self.config.deepspeed)  # noqa: F841
        #print(self.config.deepspeed)
        model_factory = self.config.model.factory(self)
        self.model = model_factory()

        see_memory_usage("after model", force=True)

        UlyssesSPAttentionHF.validate_model(
            model=self.model,
            sequence_parallel_size=self.config.sequence_parallel_size,
        )

        optimizer_factory = self.config.optimizer.factory(self)
        self.optimizer = optimizer_factory()

        scheduler_factory = self.config.scheduler.factory(self)
        self.scheduler = scheduler_factory()

        torch.distributed.barrier()
        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            args=self.config,
            lr_scheduler=self.scheduler,
            config=self.config.deepspeed,
            mpu=mpu,
        )

        see_memory_usage("after ds", force=True)

        self.checkpoint_engines = [engine(self) for engine in self.config.checkpoint_engines]

        for engine in self.checkpoint_engines:
            if engine.config.auto_resume:
                engine.load(self.model)

        self.metrics = Metrics(self)

        if self.global_rank == 0 and self.config.wandb.enable:
            # Note: wandb.init() is not type annotated so we need to use type: ignore
            self.wandb_experiment = wandb.init(  # type: ignore
                entity=self.config.wandb.entity,
                project=self.config.wandb.project,
                name=self.config.wandb.name,
                config=self.config.model_dump(),
            )

    def _set_seeds(self, seed: int) -> None:
        logger.info(f"Setting random seeds to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        set_seed(seed)

    @property
    def model_unwrapped(self):
        """Return the original model before it was wrapped by deepspeed"""
        if hasattr(self.model, "module"):
            return self.model.module
        else:
            return self.model

    @property
    def epochs(self) -> tqdm:
        """Epochs iterator."""
        return tqdm(
            range(self.epoch_idx, self.config.epochs),
            desc="Epochs",
            unit="epoch",
            disable=(self.global_rank != 0) or (self.config.train_log_iter_interval != 0),
        )

    @property
    def train_batches(self) -> tqdm:
        """Training data iterator."""
        return tqdm(
            self.train_dataloader,
            desc="Train Batches",
            unit="batch",
            disable=(self.global_rank != 0) or (self.config.train_log_iter_interval != 0),
        )

    @cached_property
    def device(self) -> torch.device:
        """Current device."""
        return torch.device(get_accelerator().device_name(self.config.local_rank))

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
        return self.config.epochs * len(self.train_dataloader) // self.config.gradient_accumulation_steps

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
        loss = self.loss(batch)
        self.model.backward(loss)

        # XXX: uncomment to compare loss exactness vs dp8-sp1
        # self.temp_losses.append(loss.item())
        # sp_world_size = 8
        # if len(self.temp_losses) == sp_world_size:
        #     avg_loss = sum(self.temp_losses) / len(self.temp_losses)
        #     print(f"{avg_loss=}")
        #     self.temp_losses = []

        # if self.config.sequence_parallel_size == 1:
        #     loss = self.loss(batch)
        #     self.model.backward(loss)

        #     # with torch.no_grad():
        #     #     # average losses since they are different on each dp rank
        #     #     losses_per_rank = torch.distributed.nn.functional.all_gather(loss)
        #     #     #print(f"LOSS {losses_per_rank=}")
        #     #     average_loss = torch.cat([l.unsqueeze(0) for l in losses_per_rank], dim=0).mean()
        #     #     #print(f"LOSS {average_loss=}")
        #     #     loss = average_loss

        # else:
        #     # sp will do backward inside sp_fwd_bwd_loss
        #     # the returned loss is already averaged across ranks and it's a float
        #     loss = self.sp_fwd_loss_bwd(batch)

        see_memory_usage("after backward", force=False)

        def maybe_item(v):
            return v.item() if torch.is_tensor(v) else v
        self.metrics.record("loss", maybe_item(loss))

        self.model.step()

        see_memory_usage("after step", force=False)
        #exit()

            # # should loss be averaged over sp sub-steps and logged as such?
            # loss = loss_aggregate / sp_world_size
            # print_rank0(f"averaged loss = {loss}")
            # #exit()



        #from deepspeed.utils import safe_get_full_grad, safe_get_full_fp32_param

        # use deepspeed global step as golden truth
        self.global_step = self.model.global_steps
        if self.global_step >= self.training_horizon:
            self.early_stop = True

        self.checkpoint()

        if self.config.exit_iteration > 0 and self.config.exit_iteration == self.global_step:
            self.early_stop = True
            logger.info(f"Hit exit iteration of {self.global_step}, ending training")


    @callback_wrapper("epoch")
    def epoch(self) -> None:
        """
        Epoch training loop. This method will be called for each epoch of
        training and iterates across batches of training data, calling the step
        method on each batch.
        """
        self.epoch_finished = False
        self.metrics.start_timer("iter")

        see_memory_usage(f"entered epoch", force=True)
        #exit()

        # enable memory history, which will add tracebacks and event history to snapshots
        if self.mem_profiler == "step":
           torch.cuda.memory._record_memory_history(max_entries=100_000)

        train_batches = self.train_batches
        if self.config.sequence_parallel_size > 1:
            from deepspeed.utils import groups
            self.sp_group = groups._get_sequence_parallel_group()
            self.sp_world_size = groups._get_sequence_parallel_world_size()
            self.sp_rank = groups._get_sequence_parallel_rank()

            train_batches = UlyssesSPDataLoaderWrapper(
                train_batches,
                sp_rank=self.sp_rank,
                sp_group=self.sp_group,
                sp_world_size=self.sp_world_size,
                device=self.device,
            )
            # this will break on epoch 2+ as it'd continue multiplying the previous value from epoch 1
            self.config.exit_iteration *= self.sp_world_size
            #self.training_horizon *= self.sp_world_size
            self.metrics.max_iter *= self.sp_world_size

        # XXX: this counter must not be reset between epochs
        self.train_batch_idx = 0
        for batch in train_batches:
            self.train_batch_idx += 1
            print_rank(f"\n\n\n\n\nITERATION: {self.train_batch_idx} ", skip=False)

            self.metrics.record("seqlen", len(batch["input_ids"][0])*self.config.sequence_parallel_size)

            see_memory_usage(f"before step", force=True)

            self.metrics.start_timer("step")
            self.step(batch)
            self.metrics.stop_timer("step")

            see_memory_usage(f"after step", force=True)

            self.metrics.restart_timer("iter")

            if (
                self.config.train_log_iter_interval != 0
                and self.train_batch_idx % self.config.train_log_iter_interval == 0
            ):
                self.metrics.print_summary()
                if (
                    self.global_rank == 0
                    and self.train_batch_idx > 1  # first iter is a massive outlier
                    and self.wandb_experiment is not None
                ):
                    self.wandb_experiment.log(
                        {k: v for k, v in self.metrics.summary_dict.items() if k != "iter"},
                        step=self.model.global_steps,
                    )

            if self.early_stop:
                break
        self.metrics.stop_timer("iter")
        self.epoch_finished = True


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
        finally:
            if self.mem_profiler == "e2e" or self.mem_profiler == "step":
                torch.cuda.memory._dump_snapshot(f"mem/mem_snapshot.{self.global_rank}.pickle")

            if self.wandb_experiment is not None:
                self.wandb_experiment.finish()

    @callback_wrapper("checkpoint")
    def checkpoint(self) -> None:
        for engine in self.checkpoint_engines:
            if engine.do_checkpoint:
                logger.info(f"Saving Checkpoint at global step: {self.global_step}.")
                engine.save(self.model)
