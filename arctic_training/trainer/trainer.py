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
from typing import Tuple

import deepspeed
import numpy as np
import torch
from deepspeed.accelerator import get_accelerator
from devtools import debug
from tqdm import tqdm
from transformers import set_seed

from arctic_training.callback.logging import post_loss_log_cb
from arctic_training.callback.mixin import CallbackMixin
from arctic_training.callback.mixin import callback_wrapper
from arctic_training.checkpoint.engine import CheckpointEngine
from arctic_training.config.trainer import TrainerConfig
from arctic_training.data.factory import DataFactory
from arctic_training.logging import logger
from arctic_training.model.factory import ModelFactory
from arctic_training.optimizer.factory import OptimizerFactory
from arctic_training.scheduler.factory import SchedulerFactory
from arctic_training.tokenizer.factory import TokenizerFactory
from arctic_training.debug import print_rank0, print_rank, exit, debug_gathered_tensor

try:
    from transformers.integrations.deepspeed import HfDeepSpeedConfig
except ImportError:
    from transformers.deepspeed import HfDeepSpeedConfig

import arctic_training.trainer.parallel_state as mpu

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
        process_group: dist.ProcessGroup
    ) -> None:

        super(UlyssesAttentionHF, self).__init__()
        self.attn = attn
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.sp_rank = dist.get_rank(process_group)
        
        self.local_seq_length = local_seq_length
        self.global_seq_length = global_seq_length
        self.batch_size = batch_size
        
        self.attn_head_size = attn_head_size
        self.attn_head_count = attn_head_count
        self.global_kv_head_count = kv_head_count

        self.local_q_head_count = attn_head_count // self.world_size
        self.local_kv_head_count = kv_head_count // self.world_size

        print_rank0(f"{self.local_kv_head_count=}")
        #exit()

        if self.attn_head_count % self.world_size != 0:
            raise ValueError(f"Attention head count {attn_head_count} is not divisible by world size {self.world_size}")
        if not (self.global_kv_head_count % self.world_size == 0 or self.world_size % self.global_kv_head_count == 0):
            raise ValueError(f"KV attention head count {self.global_kv_head_count} is not divisible by world size {self.world_size} or vice versa")

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
        # self.required_input_shape = torch.Size([local_seq_length, \
        #                                         batch_size, \
        #                                         attn_head_count, \
        #                                         attn_head_size])
        
        # [sl bs em_l]        
        self.required_context_shape = torch.Size([global_seq_length, \
                                                batch_size, \
                                                attn_head_size * attn_head_count // self.world_size])
        
    def _combine_local_sequences(self, query, key, value) -> Tuple[Tensor, Tensor, Tensor]:
        
        def combine_sequence(input, local_head_count):
            """
                expects inputs in shape: [sl_l bs hc hs]
                returns output in shape: [sl bs hc_l hs]

                local_head_count could be different for k,v vs q if it's not an MHA situation          
            """

            print_rank0('')
            print_rank0(f"combine: before reshape:  {input.shape=}")

            # [sl_l bs hc hs] -> [sl_l bs ws hc_l hs]
            input = input.reshape([self.local_seq_length, \
                                self.batch_size, \
                                self.world_size, \
                                local_head_count, \
                                self.attn_head_size])    

            print_rank0(f"combine: after reshape:   {input.shape=}")

            input = rearrange(input, 'sl_l bs ws hc_l hs -> ws sl_l bs hc_l hs').contiguous()
            print_rank0(f"combine: after rearrange: {input.shape=}")
            
            output = _DimZeroAllToAll.apply(self.process_group, input)
            print_rank0(f"combine: after all2all:   {output.shape=}")
        
            # [ws sl_l bs hc_l hs] -> [sl bs hc_l hs]
            output = output.reshape([self.global_seq_length, *output.shape[2:]]).contiguous()
            print_rank0(f"combine: after reshape:   {output.shape=}")

            # [sl bs hc_l hs]
            return output
        
        
        return combine_sequence(query, self.local_q_head_count),  combine_sequence(key, self.local_kv_head_count),  combine_sequence(value, self.local_kv_head_count)
        
    def _partition_global_sequence(self, input) -> Tensor:
        """
            expects input in shape:  [sl bs em_l] 
            returns output in shape: [sl_l bs em]
        """
        
        print_rank0(f"partition: before reshape:  {input.shape=}")

        # [sl bs em_l] -> [ws sl_l bs em_l]
        input = input.reshape([self.world_size, \
                            self.local_seq_length, \
                            self.batch_size, \
                            self.attn_head_size * self.attn_head_count // self.world_size]).contiguous()    
        
        print_rank0(f"partition: after reshape:   {input.shape=}")
        output = _DimZeroAllToAll.apply(self.process_group, input)
        print_rank0(f"partition: after all2all:   {output.shape=}")
        output = rearrange(output, 'ws sl_l bs em_l -> sl_l bs ws em_l')
        #output = rearrange(output, 'ws sl_l bs ... -> sl_l bs ws ...')
        print_rank0(f"partition: after rearrange: {output.shape=}")
            
        # [sl_l bs ws em_l] -> [sl_l bs em]
        output = output.reshape([*output.shape[:2], -1]).contiguous()
        print_rank0(f"partition: after reshape:   {output.shape=}")

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
        # UA expects:
        # [seqlen, batch_size, num_heads, head_size]
        # print(f"{query.shape=}")
        # print(f"{key.shape=}")
        # print(f"{value.shape=}")
        # print(f"{self.required_input_shape=}")   

        print_rank0(f"forward 1 {query.shape=}")
        print_rank0(f"forward 1 {key.shape=}")
        print_rank0(f"forward 1 {value.shape=}")

        query = rearrange(query, 'bs hc sl hs -> sl bs hc hs')
        key = rearrange(key,     'bs hc sl hs -> sl bs hc hs')
        value = rearrange(value, 'bs hc sl hs -> sl bs hc hs')
        
        print_rank0(f"forward 2 {query.shape=}")
        print_rank0(f"forward 2 {key.shape=}")
        print_rank0(f"forward 2 {value.shape=}")
        print_rank0(f"forward 2 {self.required_query_shape=}")
        print_rank0(f"forward 2 {self.required_key_value_shape=}")

        #print_rank0(f"{attention_mask.shape=}")
        # please don't remove the white-space vertical alignment in the error message
        assert query.shape == self.required_query_shape, \
            f"[{dist.get_rank()}]: query input tensor does not match the required shape\n             {self.required_query_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}"
        assert key.shape == value.shape == self.required_key_value_shape, \
            f"[{dist.get_rank()}]: key or value input tensor does not match the required shape\n             {self.required_key_value_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}"
        # assert query.shape == key.shape == value.shape == self.required_input_shape, \
        #     f"[{dist.get_rank()}]: One of the input tensors does not match the required shape\n             {self.required_input_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}"
        
        # expects: [sl_l bs hc hs]
        query_layer, key_layer, value_layer = self._combine_local_sequences(query, key, value)
        # returns: [sl bs hc_l hs]           

        query_layer = rearrange(query_layer, 'sl bs hc_l hs -> bs hc_l sl hs')
        key_layer = rearrange(key_layer,     'sl bs hc_l hs -> bs hc_l sl hs')
        value_layer = rearrange(value_layer, 'sl bs hc_l hs -> bs hc_l sl hs')

        print_rank0(f"{query_layer.shape=}")
        print_rank0(f"{key_layer.shape=}")
        print_rank0(f"{value_layer.shape=}")

        if attention_mask is not None:
            print_rank0(f"{attention_mask.shape=}")
            print_rank0(f"{attention_mask=}")

        # XXX: stick into the trainer object
        from deepspeed.utils import groups
        sp_group = groups._get_sequence_parallel_group()
        sp_world_size = groups._get_sequence_parallel_world_size()

        #exit()
        debug_gathered_tensor(query_layer, sp_group, name="query_layer")
        debug_gathered_tensor(key_layer, sp_group, name="key_layer")
        debug_gathered_tensor(value_layer, sp_group, name="value_layer")

        print_rank0(f"HF before real attn: {query_layer.shape=}")
        print_rank0(f"HF before real attn: {key_layer.shape=}")
        print_rank0(f"HF before real attn: {value_layer.shape=}")
        print_rank0(f"HF before real attn: {torch.norm(query_layer)=}")
        print_rank0(f"HF before real attn: {torch.norm(key_layer)=}")
        print_rank0(f"HF before real attn: {torch.norm(value_layer)=}")

        # expects: [bs hc_l sl hs]
        context_layer, attn_weights = self.attn(module, query_layer, key_layer, value_layer, attention_mask, *args, **kwargs)
        # returns [bs sl hc_l hs]

        debug_gathered_tensor(context_layer, sp_group, name="context_layer")

        print_rank0(f"HF after real attn: {context_layer.shape=}")
        print_rank0(f"HF after real attn: {torch.norm(context_layer)=}")  

        print_rank0(f"1 {context_layer.shape=}")
        # [bs sl hc_l hs] -> [sl bs hc_l hs]'
        context_layer = rearrange(context_layer, 'bs sl ... -> sl bs ...')
        print_rank0(f"2 {context_layer.shape=}")  
        context_layer = context_layer.reshape([*context_layer.shape[:2], -1])
        print_rank0(f"3 {context_layer.shape=}")  
        print_rank0(f"{self.required_context_shape=}")  

        assert context_layer.shape == self.required_context_shape, \
                    f"The context shape {context_layer.shape} is not as expected shape {self.required_context_shape}"

        # expects: [sl bs em_l]       
        output = self._partition_global_sequence(context_layer)
        # returns: [sl_l bs em]
         
        print(f"1 {output.shape=}")
        output = rearrange(output, 'sl_l bs ... -> bs sl_l ...')
        print(f"2 {output.shape=}")

        output = output.reshape([*output.shape[:2], -1])
        print(f"3 {output.shape=}")
        if attn_weights is not None:
            print_rank0(f"{attn_weights.shape=}")  

        debug_gathered_tensor(output, sp_group, name="output")
        

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


        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        print(f"{query_states.shape=}")
        print(f"{key_states.shape=}")
        print(f"{value_states.shape=}")

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        import transformers
        attention_interface: Callable = transformers.models.llama.modeling_llama.eager_attention_forward
        
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # XXX: meanwhile for consistency with non-sp testing
        #attention_interface: Callable = transformers.models.llama.modeling_llama.eager_attention_forward
        #attention_interface: Callable = transformers.integrations.flash_attention.flash_attention_forward
        attention_interface: Callable = transformers.integrations.sdpa_attention.sdpa_attention_forward

        # XXX: 
        if "ulysses" in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS["ulysses"]
            print_rank0(f"custom attention on {torch.distributed.get_rank()}")

        print_rank0(f"HF before attn: {query_states.shape=}")
        print_rank0(f"HF before attn: {key_states.shape=}")
        print_rank0(f"HF before attn: {value_states.shape=}")
        print_rank0(f"HF before attn: {torch.norm(query_states)=}")
        print_rank0(f"HF before attn: {torch.norm(key_states)=}")
        print_rank0(f"HF before attn: {torch.norm(value_states)=}")

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

        #exit()
        print_rank0(f"HF after attn: {attn_output.shape=}")
        print_rank0(f"HF after attn: {torch.norm(attn_output)=}")  
        if attn_weights is not None:
            print_rank0(f"HF after attn: {attn_weights.shape=}")
            print_rank0(f"HF after attn: {torch.norm(attn_weights)=}")  

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

class Trainer(ABC, CallbackMixin):
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

    callbacks: List[Tuple[str, Callable]] = [post_loss_log_cb]
    """
    A list of callbacks for the trainer. Callbacks are specified as tuples of a
    string indicating where the callback should be placed and a callable that
    implements the callback. Callback events for the trainer include `pre-` and
    `post-` for `init`, `train`, `epoch`, `step`, and `checkpoint`.
    """

    def __init__(self, config: TrainerConfig) -> None:
        
        # reenter()
        # self.dontexist()

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

        self._set_seeds(self.config.seed)

        tokenizer_factory = self.config.tokenizer.factory(self)
        self.tokenizer = tokenizer_factory()

        data_factory = self.config.data.factory(self)
        self.train_dataloader, self.eval_dataloader = data_factory()

        # XXX: fixme
        import torch
        print(f"MPU INIT on rank {torch.distributed.get_rank()}")
        print(f"MBS  {self.config.micro_batch_size}")
        
        mpu.initialize_model_parallel(sequence_parallel_size=self.config.sequence_parallel_size)

        import transformers.models.llama.modeling_llama
        transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttentionNew

        if mpu.get_sequence_parallel_world_size() > 1:
            from transformers import AutoConfig
            # XXX:
            model_config = AutoConfig.from_pretrained(self.config.model.name_or_path)

            #attn_implementation_real =  transformers.models.llama.modeling_llama.eager_attention_forward
            #attn_implementation_real = transformers.integrations.flash_attention.flash_attention_forward
            attn_implementation_real = transformers.integrations.sdpa_attention.sdpa_attention_forward

            #from deepspeed.sequence.layer import DistributedAttention
            uattn = UlyssesAttentionHF(
                attn=attn_implementation_real,
                local_seq_length=self.config.data.max_length // mpu.get_sequence_parallel_world_size(),
                global_seq_length=self.config.data.max_length, 
                batch_size=self.config.micro_batch_size, 
                attn_head_count=model_config.num_attention_heads,
                attn_head_size=model_config.hidden_size // model_config.num_attention_heads,
                kv_head_count=model_config.num_key_value_heads,
                process_group = mpu.get_sequence_parallel_group(),
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
                
                # XXX: for MQA key/value num of heads is different from query - currently working with MHA models to overcome this

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

            ALL_ATTENTION_FUNCTIONS["ulysses"] = uattn_wrapper

        dschf = HfDeepSpeedConfig(self.config.deepspeed)  # noqa: F841
        model_factory = self.config.model.factory(self)
        self.model = model_factory()

        optimizer_factory = self.config.optimizer.factory(self)
        self.optimizer = optimizer_factory()

        scheduler_factory = self.config.scheduler.factory(self)
        self.scheduler = scheduler_factory()

        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            args=self.config,
            lr_scheduler=self.scheduler,
            config=self.config.deepspeed,
            mpu=mpu,
        )
        

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
        
        import deepspeed.comm as dist   
        import q
        from deepspeed.utils import groups
        q(self.global_rank)
        print(f"{groups._get_sequence_parallel_group()=}")
        print(f"{groups._get_sequence_parallel_rank()=}")
        print(f"{groups._get_sequence_parallel_world_size()=}")
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
        #     print(batch)

        #self.tokenizer.chat_template = 
        
        #texts = ["this is a first very very very long prompt about", "this is a second prompt that is shorter than"]


        if 0:
            messages = [
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who answers user's questions with details and curiosity.",
                    },
                    {
                        "role": "user",
                        "content": "What are some potential applications for quantum computing?",
                    },
                ],
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who answers user's questions with details and curiosity.",
                    },
                    {
                        "role": "user",
                        "content": "What are some good ideas?",
                    },
                ],
            ]

            texts = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            batch = self.tokenizer(texts, padding="max_length", max_length=self.config.data.max_length, return_tensors='pt')

            batch["position_ids"] = torch.cat(
                [
                    torch.arange(batch["input_ids"].shape[1]).unsqueeze(0) for _ in range(batch["input_ids"].shape[0])
                    ]
                )
            #print(batch)
            #import sys; sys.exit(0)

        # print_rank0(batch)
        # exit()

        batch = {'input_ids': torch.tensor([[   529,  29989,    326,  29918,   2962,  29989,  29958,   1792,     13,  29954,   5428,    278,   1426,  29901,
          23212,   1581,    338,   5545,    553,  27797,    363,   3619,  19309,  15313,   9466,    322,   2301,  16637,
           3657,    267,  29889,    739,    884,    577,    720,    267,    322,  10208,   9100,    599,  24646,  29892,
          19912,    304,   1104,   2418,   9950,  12137,   1550,  12515,  19309,   5941,    322,   4964,  29889,     13,
          29879,    397,   1974,   5112,  25046,    313,  12574,    515,  18853,    288,   2719,  29897,   1919,  20892,
           1974,    274,   6235,    403,    313,    345,    657,    519,  10723,    511,  29909,  21408,  29892,    402,
            368,   2265,    262,    313,    345,    657,    519,  10723,  29897,  22181,    392,   2497,   2614,    504,
          28963,  17182,  29892,  22181,    392,   2497,   7498,   1182,   1458,  17182,  29892,  20892,   1974,    521,
           5095,    680,  29892,  20892,   1974,   7537,  10492,  29892,    301,    979,   1507,  29892,   2485,    265,
           1600,  29892,   5681,   3270,    324,  29889,     13,  29924,   1943,    515,   8296,    443,    999,   1312,
            288,   9258,  17182,  29892,   7137,   4244,    338,   2924,    322,   9914,    304,    278,  19309,  29889,
           7137,   4244,    338,    451,   9528,    373,  15006,  29892,   3743,    694,  23116,  28061,    470,  23895,
           9351,  29892,    338,   6446,  18655,  13956,  29889,  29871,  29896,  29900,  29900,  29995,   4768,    356,
           5105,    519,  29889,    317,    481,    265,   2164,    438,   9258,    438,    309,  29892,   3080,  13537,
           3956,   1372,  29889,   2166,   8270,    363,  18655,   1306,    550,    322,  12461,    550,  29889,     13,
           5618,    526,    777,    310,    278,  23633,    310,  22181,   1581,  17182,    363,    278,  19309,    322,
           2301,   7799,  29973,  29966,  29989,    326,  29918,    355,  29989,  29958,     13,  29966,  29989,    326,
          29918,   2962,  29989,  29958,    465,  22137,     13,  29931,    485,   1581,  17182,    338,   5545,    553,
          27797,    363,   3619,  19309,  15313,   9466,    322,   2301,  16637,   3657,    267,  29889,    739,   6911,
            304,    577,  10323,    322,  26681,    599,  24646,  29892,   6911,    304,   1104,   2418,   9950,  12137,
           1550,  12515,  19309,   5941,    322,   4964,  19423,  29989,    326,  29918,    355,  29989,  29958,     13,
          29966,  29989,    326,  29918,   2962,  29989,  29958,   1792,     13,   6028,    366,   3113,   3867,    592,
            411,    278,   1051,    310,   2348,   1127,  10070,    297,    278,   7137,   4244,  29559,   5276,    297,
            278,   1426,  29973,  29966,  29989,    326,  29918,    355,  29989,  29958,     13,  29966,  29989,    326,
          29918,   2962,  29989,  29958,    465,  22137,     13,   1576,   2348,   1127,  10070,    297,   7137,   4244,
          29559,   5276,    297,    278,   1426,    526,    317,    481,    265,   2164,    438,   9258,    438,    309,
            322,   3080,  13537,   3956,   1372,  19423,  29989,    326,  29918,    355,  29989,  29958,     13,  29966,
          29989,    326,  29918,   2962,  29989,  29958,   1792,     13,   6028,    366,   3113,   2649,    592,    565,
           7137,   4244,  29559,    338,  13907,    363,  12461,    550,  29973,  29966,  29989,    326,  29918,    355,
          29989,  29958,     13,  29966,  29989,    326,  29918,   2962,  29989,  29958,    465,  22137,     13,   8241,
          29892,   7137,   4244,  29559,    338,  13907,    363,  12461,    550,  29889,    450,   1426,   9479,  26649,
            393,    372,    338,   6446,  18655,  13956,    322,   3743,    694,  13019,  29899,   6707,   2348,   1127,
          10070,  19423,  29989,    326,  29918,    355,  29989,  29958,     13, 128009, 128009, 128009, 128009, 128009,
         128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
         128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
         128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
         128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009],
        [   529,  29989,    326,  29918,   2962,  29989,  29958,   1792,     13,   4013,    338,    263,  12983,   1813,
            363,    317,   2518,  14808,  29892,    607,   2794,    445,   2022,    338,    451,   5279,    373,    445,
           3268,  29889,   1334,    437,   4368,    773,    278,   8492,   2400,    304,   1284,    317,   2518,  14808,
          29889,     13,   3492,    526,   6493,    292,    278,  12983,   1813,    363,    317,   2518,  14808,  29889,
            910,   1813,    338,   1244,   1363,   4856,   1304,   1749,  12983,  19725,    304,   1106,    363,    317,
           2518,  14808,  29889,   1334,   2825,    445,   1813,   6336,    297,  26926,    317,   2518,  14808,    723,
           1284,    372,  29889,    960,    366,    526,    451,    317,   2518,  14808,  29892,    541,    526,    385,
            394,   1227,  29875,    310,   1522,   5309,   6115,   5057,   4523,  29892,   6036,    373,    445,   3268,
            363,   3889,   1286,  29889,     13,   5328,    471,    445,  12983,   1813,   2825,    363,    317,   2518,
          14808,    322,   2020,    338,    372,   1244,  29973,  29966,  29989,    326,  29918,    355,  29989,  29958,
             13,  29966,  29989,    326,  29918,   2962,  29989,  29958,    465,  22137,     13,   4013,  12983,   1813,
            471,   2825,   6336,    491,    278,   4700,  29915,  29879,   1788,    746,   4856,  17371,    363,    317,
           2518,  14808,    541,   1183,   1258,    451,    505,    385,   5923,   8722,    373,    278,   3268,  29889,
           8011,   6437,    338,    304,   1044,    408,    263,  13201,  25325,   1813,    297,  26926,    393,    317,
           2518,  14808,    674,   2041,   4822,    372,    322,   1653,    263,   8722,    373,    278,   3268,  19423,
          29989,    326,  29918,    355,  29989,  29958,     13,  29966,  29989,    326,  29918,   2962,  29989,  29958,
           1792,     13,   6028,    366,   3113,   2649,    592,    565,    727,    526,    738,    916,   5837,    304,
           2740,    363,    317,   2518,  14808,  12435,    515,    278,   8492,   5276,    373,    445,   1813,  29973,
          29966,  29989,    326,  29918,    355,  29989,  29958,     13,  29966,  29989,    326,  29918,   2962,  29989,
          29958,    465,  22137,     13,  29902,    437,    451,    505,   2130,    304,   2702,   2472,   1048,    269,
           2518,   1886,    262,  29915,  29879,    988,  12717,  29879,    470,   6958,   2472,  29889,   2398,  29892,
           1244,    526,    777,   2498,  10529,    363,   5837,    304,   2740,    363,   4856,  29901,     13,     13,
          29896,  29889,   4803,   5264,   5745,  21796,    763,  18335,    470,   9024,    262,    304,   2740,    363,
            269,   2518,   1886,    262,  29889,     13,     13,  29906,  29889,  22387,    367,   5309,   6115,   1880,
           3762,  29915,  29879,    394,   1227,  29875,   8034,    470,   2613,   2838,    272,    304,    297,   1548,
            565,    896,    505,    738,   2472,   1048,    269,   2518,   1886,    262,    470,    902,    988,  12717,
          29879,  29889,     13,     13,  29941,  29889,   4803,   7395,   2740,  24000,    763,   5386,    304,   2740,
            363,    269,   2518,   1886,    262,    322,    738,   8018,   2472,   1316,    408,    902,   4423,    322,
          26818,  29889,     13,     13,  29946,  29889,    830,    496,    714,    304,   5478,    950,  19395,   2925,
            470,   7875,   1058,   1122,    505,   2472,   1048,    269,   2518,   1886,    262,  29915,  29879,    988,
          12717,  29879,  29889,     13,     13,  29945,  29889,  22310,    675,  12530,   2305,   2740,   5786,    393,
           4266,    675,    297,   9138,   6958,   2472,    363,  15724,  19423,  29989,    326,  29918,    355,  29989,
          29958,     13, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
         128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
         128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
         128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009,
         128009, 128009, 128009, 128009, 128009, 128009, 128009, 128009]]), 'labels': torch.tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100, 29931,   485,  1581, 17182,   338,  5545,   553, 27797,   363,  3619, 19309,
         15313,  9466,   322,  2301, 16637,  3657,   267, 29889,   739,  6911,   304,   577, 10323,   322, 26681,   599,
         24646, 29892,  6911,   304,  1104,  2418,  9950, 12137,  1550, 12515, 19309,  5941,   322,  4964, 19423,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  1576,  2348,  1127, 10070,   297,  7137,  4244, 29559,  5276,
           297,   278,  1426,   526,   317,   481,   265,  2164,   438,  9258,   438,   309,   322,  3080, 13537,  3956,
          1372, 19423,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  8241, 29892,  7137,  4244, 29559,   338, 13907,   363, 12461,   550, 29889,   450,  1426,
          9479, 26649,   393,   372,   338,  6446, 18655, 13956,   322,  3743,   694, 13019, 29899,  6707,  2348,  1127,
         10070, 19423,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100],
        [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  4013, 12983,  1813,   471,  2825,  6336,   491,   278,  4700,
         29915, 29879,  1788,   746,  4856, 17371,   363,   317,  2518, 14808,   541,  1183,  1258,   451,   505,   385,
          5923,  8722,   373,   278,  3268, 29889,  8011,  6437,   338,   304,  1044,   408,   263, 13201, 25325,  1813,
           297, 26926,   393,   317,  2518, 14808,   674,  2041,  4822,   372,   322,  1653,   263,  8722,   373,   278,
          3268, 19423,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 29902,   437,
           451,   505,  2130,   304,  2702,  2472,  1048,   269,  2518,  1886,   262, 29915, 29879,   988, 12717, 29879,
           470,  6958,  2472, 29889,  2398, 29892,  1244,   526,   777,  2498, 10529,   363,  5837,   304,  2740,   363,
          4856, 29901,    13,    13, 29896, 29889,  4803,  5264,  5745, 21796,   763, 18335,   470,  9024,   262,   304,
          2740,   363,   269,  2518,  1886,   262, 29889,    13,    13, 29906, 29889, 22387,   367,  5309,  6115,  1880,
          3762, 29915, 29879,   394,  1227, 29875,  8034,   470,  2613,  2838,   272,   304,   297,  1548,   565,   896,
           505,   738,  2472,  1048,   269,  2518,  1886,   262,   470,   902,   988, 12717, 29879, 29889,    13,    13,
         29941, 29889,  4803,  7395,  2740, 24000,   763,  5386,   304,  2740,   363,   269,  2518,  1886,   262,   322,
           738,  8018,  2472,  1316,   408,   902,  4423,   322, 26818, 29889,    13,    13, 29946, 29889,   830,   496,
           714,   304,  5478,   950, 19395,  2925,   470,  7875,  1058,  1122,   505,  2472,  1048,   269,  2518,  1886,
           262, 29915, 29879,   988, 12717, 29879, 29889,    13,    13, 29945, 29889, 22310,   675, 12530,  2305,  2740,
          5786,   393,  4266,   675,   297,  9138,  6958,  2472,   363, 15724, 19423,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100]]), 'position_ids': torch.tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
          22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
          44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
          66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
          88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
         110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
         132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
         154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
         176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
         198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
         220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
         242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
         264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
         286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307,
         308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329,
         330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
         352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373,
         374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
         396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
         418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
         440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461,
         462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483,
         484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505,
         506, 507, 508, 509, 510, 511],
        [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
          22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
          44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
          66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
          88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
         110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
         132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
         154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
         176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
         198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
         220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
         242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
         264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
         286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307,
         308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329,
         330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
         352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373,
         374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
         396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
         418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
         440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461,
         462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483,
         484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505,
         506, 507, 508, 509, 510, 511]])}


        # tokenized with Felladrin/Llama-160M-Chat-v1
        if 0:
            batch =  {'input_ids': torch.tensor([[  529, 29989,   326, 29918,  2962, 29989, 29958,  1792,    13, 29899, 18555,   781,   950,  2875, 27428, 29966,
         29989,   326, 29918,   355, 29989, 29958,    13, 29966, 29989,   326, 29918,  2962, 29989, 29958,   465, 22137,
            13,  2928,   295,   781,   950,  2875, 27428, 14637,   304,   278, 11706, 10462,   393, 15724,   470, 14582,
           505,   304, 29192,   671,   310,  1009,   907,   800,   470, 24233,   800, 29889,   910,  7805, 10462,   304,
          2373,  1237, 29892,  1020,  2310, 17862, 29892,  3509,  1266, 29879, 29892,   322, 11302, 22183,  1372, 29889,
         18555,   781,   950,  2875, 27428,   338,  4100,  1363,   372,  6511,   907,  4097,   322, 24233,  4097,   304,
         21665,   515,  1009,   664,   322, 28057,  4045,   515,   773, 29892,   269,  7807, 29892,   470,  2600, 11407,
           515,  1009,   907,   800,  1728, 10751, 29889,   739,   884, 18443,   267, 24233,   362,   322,   907, 28157,
         29892,   408, 15724,   322, 14582,   508, 13258,   931,   322,  7788,   964, 14338,   716,  9316,   322,  7014,
         13797,   393,   896,   674,   367,  6364,   491,  4307, 19423, 29989,   326, 29918,   355, 29989, 29958,    13,
         29966, 29989,   326, 29918,  2962, 29989, 29958,  1792,    13,  6028,   366,  3867,   901,  2702,  6455,   310,
           825,   338,  5545, 29762,  2875,   322,   825,   338,   451, 29973, 29966, 29989,   326, 29918,   355, 29989,
         29958,    13, 29966, 29989,   326, 29918,  2962, 29989, 29958,   465, 22137,    13, 29903,   545, 29892,   306,
           508,  2367,   777,  6455,   310,   825,   338,  5545,   408, 29762,  2875,   322,   825,   338,   451, 29889,
         18555,   781,   950,  2875,  7805, 29901,    13,    13, 29899,  4121,  1237, 29901, 29192, 10462, 16896,   304,
         11817,   943,   363,   716,   322,  5407,  9316,   470, 10174,    13,    13, 29899,  3201,  2310, 17862, 29901,
          8359,   573, 18906,   470, 15072,  1304,   304, 12439,   322, 20820,  9316,   470,  5786,    13,    13, 29899,
         14187,  1266, 29879, 29901, 29192, 10462, 16896,   304,   907,  4097,   310,  2441,  1736,  1316,   408,  8277,
         29892,  4696, 29892,   322,  7047,    13,    13, 29899, 27226, 22183,  1372, 29901, 24332,   616,  2472,   393,
          4076, 14582,   263,  5100,  3321, 10631, 29892,  1316,   408,  7035, 26760,   470, 12012,  3864, 10174, 29871,
            13,    13,  5618,   338,   451,  5545, 29762,  2875,  7805, 29901,    13,    13, 29899, 13001,   294, 29901,
         22001,   470, 25841,  7432,  2609,   367,  6364,   491, 29762,  2875, 10462,  6521,   896,   526,  7232,   397,
          1000,   297,   263, 18806,  1821,   883,  1316,   408,   263, 22267,   470,  2373,   296,  2280,    13,    13,
         29899, 26748, 29879, 29901,  2472,   393,   338,  3619,  7134,   470,   297,   278,   970,  5354,   338,   451,
          6364,   491, 29762,  2875, 10462,    13,    13, 29899, 14706,   322, 17735, 29901,  6521,  1304,   297, 10296,
           411,   263,  8359,   573,  5829,   470,  2874, 29892,  2983,   322, 17735,  2609,   367,  6364,   408,  1020,
          2310, 17862,    13,    13, 29899, 13976,   297,   278,   970,  5354, 29901,  1736,   393,   505,  1518,  2859,
          3509,  1266, 13047,   470,  2360,   750,  3509,  1266, 13047,   526,   297,   278,   970,  5354,   322,   508,
           367,  1304,  1728, 10751,   515,   278,   907,  1061, 19423, 29989,   326, 29918,   355, 29989, 29958,    13,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2],
        [  529, 29989,   326, 29918,  2962, 29989, 29958,  1792,    13, 29954,  5428,   278,  1426, 29901,  3532,  1074,
           599,  5457, 29915, 29879, 26412, 29877,   476, 23213,  9316, 29889,    13, 29898, 19499,  1302,   535,   329,
         29892,   607,   338,  5722,  1711,  5545,   263,  5447, 18254,   491,   278,   383,  7698,   467,    13, 11548,
           274,  1241, 13848,   387,   279, 29892, 17455, 21242,   784, 11541, 29892,   289,  5086,   269,  8887, 29892,
           330,  5621, 29892,   274,  2559,   314,   265, 29892,   413,   359,  2276, 15795, 29889,    13,  7566,  2715,
           373, 29901, 27822, 29871, 29896, 29906,  4779, 29892, 29871, 29906, 29900, 29896, 29947, 29889,    13, 29954,
          5621, 21046,   505,  2337,  1063,   263, 25448,   310,  7903, 29889,  1932,   306,  4312,   304,   748,  3144,
          6935,  3889,   445,   338,  1554,   306,  2714,   306, 29915, 29881,  3052, 29991,  2398, 29892,  1438, 12773,
           708,  4964, 21046,   901,  1135,  5445,   393,  1780, 29889,  5282, 18639,   263,  1589, 11356, 29889,    13,
          2831,  3144,  6935,  3889, 21046,  1438,   892,  2289,  2107, 29889,   306,  2289,   763,  6062,   465,   267,
           322,  1438,  2833,   304,   505,   263,  6023,   310,   330,  5621,   884,   607,  3732,   963,  2289,  1781,
           304,   592, 29889,   319,  1407,  7575, 17140,   304,   738,   592,   284, 29889,    13, 29902,   763,  4359,
          3099,   393,   756,   330,  5621,   297,   372,  6824, 10750,   465,   267,   338,  1790, 25448,   577,   304,
          1284,   963,  4208, 11827,   304,   367,   263,  2289,  4266,  7539, 29889,  2811, 15649,  1449, 21004,    13,
          2831,  3144,  6935,  3889, 21046,  1438,   892,  2289,  2107, 29889,   306,  2289,   619,  6317,    13, 17506,
           599,   310,  5457, 29915, 29879, 26412, 29877,   476, 23213,  9316,  3144,  6935, 29899,  9021, 29973, 29966,
         29989,   326, 29918,   355, 29989, 29958,    13, 29966, 29989,   326, 29918,  2962, 29989, 29958,   465, 22137,
            13,  3112,   338,   451,  6790,  3692,   599,   310,  5457, 29915, 29879, 26412, 29877,   476, 23213,  9316,
           526,  3144,  6935, 29899,  9021, 29889,   450,  1426,   871, 26649,   393,   278,   330,  5621, 21046,  5276,
           526,  3144,  6935, 29899,  9021, 19423, 29989,   326, 29918,   355, 29989, 29958,    13, 29966, 29989,   326,
         29918,  2962, 29989, 29958,  1792,    13,  6028,   366,  2649,   592,   825,  2635,   278,   330,  5621, 21046,
           892,  2715,   304,  5457, 29915, 29879, 26412, 29877,   476, 23213,  9316, 29973, 29966, 29989,   326, 29918,
           355, 29989, 29958,    13, 29966, 29989,   326, 29918,  2962, 29989, 29958,   465, 22137,    13,  8241, 29892,
           278,  1426,  5922,   393,   278,   330,  5621, 21046,   892,  2715,   373, 27822, 29892, 29871, 29896, 29906,
          4779, 29871, 29906, 29900, 29896, 29947, 19423, 29989,   326, 29918,   355, 29989, 29958,    13,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2]]), 'labels': torch.tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  2928,   295,   781,   950,  2875, 27428, 14637,   304,   278, 11706, 10462,   393, 15724,   470, 14582,
           505,   304, 29192,   671,   310,  1009,   907,   800,   470, 24233,   800, 29889,   910,  7805, 10462,   304,
          2373,  1237, 29892,  1020,  2310, 17862, 29892,  3509,  1266, 29879, 29892,   322, 11302, 22183,  1372, 29889,
         18555,   781,   950,  2875, 27428,   338,  4100,  1363,   372,  6511,   907,  4097,   322, 24233,  4097,   304,
         21665,   515,  1009,   664,   322, 28057,  4045,   515,   773, 29892,   269,  7807, 29892,   470,  2600, 11407,
           515,  1009,   907,   800,  1728, 10751, 29889,   739,   884, 18443,   267, 24233,   362,   322,   907, 28157,
         29892,   408, 15724,   322, 14582,   508, 13258,   931,   322,  7788,   964, 14338,   716,  9316,   322,  7014,
         13797,   393,   896,   674,   367,  6364,   491,  4307, 19423,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 29903,   545, 29892,   306,
           508,  2367,   777,  6455,   310,   825,   338,  5545,   408, 29762,  2875,   322,   825,   338,   451, 29889,
         18555,   781,   950,  2875,  7805, 29901,    13,    13, 29899,  4121,  1237, 29901, 29192, 10462, 16896,   304,
         11817,   943,   363,   716,   322,  5407,  9316,   470, 10174,    13,    13, 29899,  3201,  2310, 17862, 29901,
          8359,   573, 18906,   470, 15072,  1304,   304, 12439,   322, 20820,  9316,   470,  5786,    13,    13, 29899,
         14187,  1266, 29879, 29901, 29192, 10462, 16896,   304,   907,  4097,   310,  2441,  1736,  1316,   408,  8277,
         29892,  4696, 29892,   322,  7047,    13,    13, 29899, 27226, 22183,  1372, 29901, 24332,   616,  2472,   393,
          4076, 14582,   263,  5100,  3321, 10631, 29892,  1316,   408,  7035, 26760,   470, 12012,  3864, 10174, 29871,
            13,    13,  5618,   338,   451,  5545, 29762,  2875,  7805, 29901,    13,    13, 29899, 13001,   294, 29901,
         22001,   470, 25841,  7432,  2609,   367,  6364,   491, 29762,  2875, 10462,  6521,   896,   526,  7232,   397,
          1000,   297,   263, 18806,  1821,   883,  1316,   408,   263, 22267,   470,  2373,   296,  2280,    13,    13,
         29899, 26748, 29879, 29901,  2472,   393,   338,  3619,  7134,   470,   297,   278,   970,  5354,   338,   451,
          6364,   491, 29762,  2875, 10462,    13,    13, 29899, 14706,   322, 17735, 29901,  6521,  1304,   297, 10296,
           411,   263,  8359,   573,  5829,   470,  2874, 29892,  2983,   322, 17735,  2609,   367,  6364,   408,  1020,
          2310, 17862,    13,    13, 29899, 13976,   297,   278,   970,  5354, 29901,  1736,   393,   505,  1518,  2859,
          3509,  1266, 13047,   470,  2360,   750,  3509,  1266, 13047,   526,   297,   278,   970,  5354,   322,   508,
           367,  1304,  1728, 10751,   515,   278,   907,  1061, 19423,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100],
        [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  3112,   338,   451,  6790,  3692,   599,   310,  5457, 29915, 29879, 26412, 29877,   476, 23213,  9316,
           526,  3144,  6935, 29899,  9021, 29889,   450,  1426,   871, 26649,   393,   278,   330,  5621, 21046,  5276,
           526,  3144,  6935, 29899,  9021, 19423,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  8241, 29892,
           278,  1426,  5922,   393,   278,   330,  5621, 21046,   892,  2715,   373, 27822, 29892, 29871, 29896, 29906,
          4779, 29871, 29906, 29900, 29896, 29947, 19423,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100]]), 'position_ids': torch.tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
          22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
          44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
          66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
          88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
         110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
         132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
         154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
         176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
         198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
         220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
         242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
         264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
         286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307,
         308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329,
         330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
         352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373,
         374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
         396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
         418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
         440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461,
         462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483,
         484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505,
         506, 507, 508, 509, 510, 511],
        [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
          22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
          44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
          66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
          88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
         110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,
         132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
         154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
         176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
         198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
         220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
         242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
         264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
         286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307,
         308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329,
         330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351,
         352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373,
         374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
         396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417,
         418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439,
         440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461,
         462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483,
         484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505,
         506, 507, 508, 509, 510, 511]])}

        # XXX: important
        # batch["attention_mask"] = torch.ne(batch["input_ids"], 2).int()

        # 2D to 4D with full seqlen to get the attention mask right
        # XXX: hardcoded for Llama for now

        # past_seen_tokens = 0 # XXX: ? what's the 2nd iteration value
        # inputs_embeds = self.model.model.embed_tokens(batch["input_ids"].to(self.device))
        # cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        # attention_mask_4D = self.model.model._update_causal_mask(
        #     attention_mask=batch["attention_mask"], 
        #     input_tensor=inputs_embeds, 
        #     cache_position=cache_position, 
        #     past_key_values=None,
        #     output_attentions=False,
        # )
        # batch["attention_mask"] = attention_mask_4D

        for k in batch.keys():
            print_rank0(f"before sp {k}: {batch[k].shape=}")
            print_rank0(f"before sp {k}: {batch[k]=}")
        #import sys; sys.exit(0)



        if 1:
            # XXX: probably need to do padding so that all sequence chunks are the same?!
            import math
            print(f"{len(batch['input_ids'][0])=}")
            #print(f"{len(batch['input_ids'][1])=}")
            #seq_length = len(batch["input_ids"][0])
            seq_length = self.config.data.max_length
            
            sp_world_size = groups._get_sequence_parallel_world_size()
            sp_rank = groups._get_sequence_parallel_rank()
            chunk_len = math.ceil(seq_length / sp_world_size)
            print(f"{seq_length=}")
            print(f"{chunk_len=}")

            #import sys; sys.exit(0)

            for k in batch.keys():
                if sp_world_size > 1 and k in ["input_ids", "position_ids"]: # , "labels"]:
                #if sp_world_size > 1 and k in ["input_ids"]:
                    batch[k] = batch[k][:, chunk_len*sp_rank:chunk_len*(sp_rank+1)].to(self.device)
                else:
                    batch[k] = batch[k].to(self.device)
                print(f"{k} {batch[k].shape=}")
            #import sys; sys.exit(0)

            #outputs = generate(batch, do_sample=False, max_new_tokens=1)   
            #print(f"RANK {self.global_rank}: GENERATED: [{outputs[0]['generated_text']}]")

            #print_rank0(f'{batch["attention_mask"]=}')
            #print_rank0(f'{batch["attention_mask"].shape=}')

            # XXX: restore attention_mask and when doing so need to chunk it along with all other fields in the batch, like input_ids 
            #del batch["attention_mask"]

        # if 0:
        #     outputs = self.model.generate(**batch, do_sample=False, max_new_tokens=1)        
        #     print(outputs)
        #     decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #     decoded_last_token0 = self.tokenizer.batch_decode([outputs[0][-1:]], skip_special_tokens=True)[0]
        #     decoded_last_token1 = self.tokenizer.batch_decode([outputs[1][-1:]], skip_special_tokens=True)[0]
        #     #if self.global_rank == 0:
            
        #     dist.barrier()
        #     # chunk = decoded[0][-100:].replace('\n',' ')
        #     # print(f"RANK {self.global_rank}: GENERATED: [{chunk}]")
        #     chunk = decoded[0].replace('\n',' ')
        #     print(f"RANK {self.global_rank}: GENERATED: [{chunk}]")
        #     print(f"RANK {self.global_rank}: NEW TOKEN[0]: [{decoded_last_token0}]")
        #     print(f"RANK {self.global_rank}: NEW TOKEN[1]: [{decoded_last_token1}]")
        #     # chunk = decoded[0][-seq_length:].replace('\n',' ')
        #     # print(f"RANK {self.global_rank}: GENERATED: [{chunk}]")

        #     #dist.barrier()
        #     #chunk = decoded[1].replace('\n',' ')
        #     #print(f"RANK {self.global_rank}: GENERATED: [{chunk}]")
            
        #     # expected 1 generated character:
        #     # 0: o
        #     # 1: .

        #     import sys; sys.exit(0)

        if 0:
            del batch["attention_mask"]

        for k in batch.keys():
            print_rank0(f"after sp: {k}: {batch[k].shape=}")
            print_rank0(f"after sp: {k}: {batch[k]=}")
        self.model.train()

        if sp_world_size > 1:
            loss = self.loss_sp(batch)
        else:
            loss = self.loss(batch)

        #print_rank(f"{self.train_batch_idx}: {loss.grad=}")
        print_rank(f"{self.train_batch_idx}: {loss.requires_grad=}")
        print_rank(f"{self.train_batch_idx}: {loss=}")
        exit()

        self.model.backward(loss)

        # #w = self.model.module.model.layers[0].self_attn.q_proj.weight
        # w = self.model.module.lm_head.weight
        from deepspeed.utils import safe_get_full_grad
        print_rank(f"{torch.norm(safe_get_full_grad(self.model.module.lm_head.weight))=}")
        print_rank(f"{torch.norm(safe_get_full_grad(self.model.module.model.layers[0].self_attn.q_proj.weight))=}")

        # for n,p in self.model.module.named_parameters():
        #     if p.requires_grad:
        #         print_rank(f"{n}: {p.numel()}")
            
        #exit()
        self.model.step()

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

    @callback_wrapper("epoch")
    def epoch(self) -> None:
        """
        Epoch training loop. This method will be called for each epoch of
        training and iterates across batches of training data, calling the step
        method on each batch.
        """
        self.train_batch_idx = 0
        for batch in self.train_batches:
            self.train_batch_idx += 1

            self.step(batch)
            if self.early_stop:
                break

    @callback_wrapper("train")
    def train(self) -> None:
        """
        Main training loop. Calls the epoch method for each epoch of training.
        """
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
