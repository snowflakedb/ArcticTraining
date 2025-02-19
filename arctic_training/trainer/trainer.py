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
from arctic_training.debug import print_rank0, print_rank

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
        em    = embedding / hidden size
        em_l  = embedding / hidden size local   

    Arguments:
        attn: normal attention implementation from transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS
        local_seq_length (int): local sequence length per GPU,
        global_seq_length (int): actual sequence length,
        batch_size (int): batch size,
        attn_head_size (int): size of each attention head,
        attn_head_count (int): total number of attention heads,        
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
        process_group: dist.ProcessGroup
    ) -> None:

        super(UlyssesAttentionHF, self).__init__()
        self.attn = attn
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        
        assert attn_head_count % self.world_size == 0, f"Attention head count {attn_head_count} is not divisible by world size {self.world_size}"

        # XXX: add more constraints (some might have to go outside of this module or change the API to add more arguments if they are needed here, or perhaps add a special class method that validates the outside things)
        # - MQA/GQA: SP size <= kv_head_count (get via sp process group world size)
        # - sl is divisible by SP
        # - more?
        
        self.local_seq_length = local_seq_length
        self.global_seq_length = global_seq_length
        self.batch_size = batch_size
        
        self.attn_head_size = attn_head_size
        self.attn_head_count = attn_head_count
        
        # [sl_l bs hc hs]
        self.required_input_shape = torch.Size([local_seq_length, \
                                                batch_size, \
                                                attn_head_count, \
                                                attn_head_size])
        
        # # [bs hc_l sl hs]
        # self.required_input_shape_for_core_attn = torch.Size([batch_size, \
        #                                                       attn_head_count, \
        #                                                       global_seq_length, \
        #                                                       attn_head_size])



        # [sl bs em_l]        
        self.required_context_shape = torch.Size([global_seq_length, \
                                                batch_size, \
                                                attn_head_size * attn_head_count // self.world_size])
        
    def _combine_local_sequences(self, query, key, value) -> Tuple[Tensor, Tensor, Tensor]:
        
        def combine_sequence(input):
            """
                expects inputs in shape: [sl_l bs hc hs]
                returns output in shape: [sl bs hc_l hs]           
            """

            print_rank0('')
            print_rank0(f"combine: before reshape:  {input.shape=}")

            # [sl_l bs hc hs] -> [sl_l bs ws hc_l hs]
            input = input.reshape([self.local_seq_length, \
                                self.batch_size, \
                                self.world_size, \
                                self.attn_head_count // self.world_size, \
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
        
        return combine_sequence(query), combine_sequence(key), combine_sequence(value)
        
    def _partition_global_sequence(self, input) -> Tensor:
        """
            expects input in shape:  [sl bs hs*hc_l] 
            returns output in shape: 
        """
        
        print_rank0(f"partition: before reshape:  {input.shape=}")

        # [sl, bs, hs*hc] -> [ws sl_l bs hs_l]
        input = input.reshape([self.world_size, \
                            self.local_seq_length, \
                            self.batch_size, \
                            self.attn_head_size * self.attn_head_count // self.world_size]).contiguous()    
        
        print_rank0(f"partition: after reshape:   {input.shape=}")
        output = _DimZeroAllToAll.apply(self.process_group, input)
        print_rank0(f"partition: after all2all:   {output.shape=}")
        output = rearrange(output, 'ws sl_l bs hs_l -> sl_l bs ws hs_l')
        print_rank0(f"partition: after rearrange: {output.shape=}")
            
        # [sl_l bs ws hs_l] -> [sl_l bs ws*hs_l]
        output = output.reshape([*output.shape[:2], -1]).contiguous()
        print_rank0(f"partition: after reshape:   {output.shape=}")

        # [sl_l bs ws*hs_l]
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
        print_rank0(f"forward 2 {self.required_input_shape=}")

        #print_rank0(f"{attention_mask.shape=}")
        # please don't remove the white-space vertical alignment in the error message
        assert query.shape == key.shape == value.shape == self.required_input_shape, \
            f"[{dist.get_rank()}]: One of the input tensors does not match the required shape\n             {self.required_input_shape}:\n {query.shape=}\n   {key.shape=}\n {value.shape=}"
        
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

        print_rank0(f"HF before real attn: {query_layer.shape=}")
        print_rank0(f"HF before real attn: {key_layer.shape=}")
        print_rank0(f"HF before real attn: {value_layer.shape=}")
        print_rank0(f"HF before real attn: {torch.norm(query_layer)=}")
        print_rank0(f"HF before real attn: {torch.norm(key_layer)=}")
        print_rank0(f"HF before real attn: {torch.norm(value_layer)=}")

        # this check I added is wrong I think
        # # please don't remove the white-space vertical alignment in the error message
        # assert query_layer.shape == key_layer.shape == value_layer.shape == self.required_input_shape_for_core_attn, \
        #     f"[{dist.get_rank()}]: One of the input tensors does not match the required_input_shape_for_core_attn\n                   {self.required_input_shape_for_core_attn}:\n {query_layer.shape=}\n   {key_layer.shape=}\n {value_layer.shape=}"

        # expects: [bs hc_l sl hs]
        context_layer, attn_weights = self.attn(module, query_layer, key_layer, value_layer, attention_mask, *args, **kwargs)
        # returns [bs sl hc_l hs]

        print_rank0(f"HF after real attn: {context_layer.shape=}")
        print_rank0(f"HF after real attn: {torch.norm(context_layer)=}")  

        print_rank0(f"1 {context_layer.shape=}")  
        context_layer = rearrange(context_layer, 'bs sl hc_l hs -> sl bs hs hc_l')
        print_rank0(f"2 {context_layer.shape=}")  
        context_layer = context_layer.reshape([*context_layer.shape[:2], -1])
        print_rank0(f"3 {context_layer.shape=}")  
        print_rank0(f"{self.required_context_shape=}")  

        assert context_layer.shape == self.required_context_shape, \
                    f"The context shape {context_layer.shape} is not as expected shape {self.required_context_shape}"

        # expects: [sl bs hs*hc_l]       
        output = self._partition_global_sequence(context_layer)
        # returns: [sl_l bs ws*hs_l]
         
        print(f"1 {output.shape=}")
        output = rearrange(output, 'sl_l bs ... -> bs sl_l ...')
        print(f"2 {output.shape=}")

        #output = output.reshape([*output.shape[:2], ]))
        #print_rank0(f"{attn_weights.shape=}")  



        # expects [bs sl hc hs]
        return output, attn_weights

# class UlyssesAttentionHF(UlyssesAttentionHF):
#     def forward(self,
#         module: torch.nn.Module,        
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         *args,
#         **kwargs,
#     ) -> Tuple[torch.Tensor, None]:
#         attn_output = super().forward(
#             query=query,
#             key=key,
#             value=value,
#             *args,
#             **kwargs
#         )
#         return attn_output, None


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
        attention_interface: Callable = transformers.integrations.flash_attention.flash_attention_forward

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
        if attn_weights is not None:
            print_rank0(f"HF after attn: {attn_weights.shape=}")
            print_rank0(f"HF after attn: {torch.norm(attn_weights)=}")  


        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
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

        from transformers import AutoConfig
        # XXX:
        model_config = AutoConfig.from_pretrained(self.config.model.name_or_path)

        import transformers.models.llama.modeling_llama
        transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttentionNew

        #attn_implementation_real =  transformers.models.llama.modeling_llama.eager_attention_forward
        attn_implementation_real = transformers.integrations.flash_attention.flash_attention_forward

        #from deepspeed.sequence.layer import DistributedAttention
        uattn = UlyssesAttentionHF(
            attn=attn_implementation_real,
            local_seq_length=self.config.data.max_length // mpu.get_sequence_parallel_world_size(),
            global_seq_length=self.config.data.max_length, 
            batch_size=self.config.micro_batch_size, 
            attn_head_count=model_config.num_attention_heads,
            attn_head_size=model_config.hidden_size // model_config.num_attention_heads, 
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

        if mpu.get_sequence_parallel_world_size() > 1:
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




        #print(batch)

        #from transformers import pipeline

        #generate = pipeline("text-generation", "Felladrin/Llama-160M-Chat-v1")

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

            print(batch)
            #import sys; sys.exit(0)

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

        if 0:
            outputs = self.model.generate(**batch, do_sample=False, max_new_tokens=1)        
            print(outputs)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_last_token0 = self.tokenizer.batch_decode([outputs[0][-1:]], skip_special_tokens=True)[0]
            decoded_last_token1 = self.tokenizer.batch_decode([outputs[1][-1:]], skip_special_tokens=True)[0]
            #if self.global_rank == 0:
            
            dist.barrier()
            # chunk = decoded[0][-100:].replace('\n',' ')
            # print(f"RANK {self.global_rank}: GENERATED: [{chunk}]")
            chunk = decoded[0].replace('\n',' ')
            print(f"RANK {self.global_rank}: GENERATED: [{chunk}]")
            print(f"RANK {self.global_rank}: NEW TOKEN[0]: [{decoded_last_token0}]")
            print(f"RANK {self.global_rank}: NEW TOKEN[1]: [{decoded_last_token1}]")
            # chunk = decoded[0][-seq_length:].replace('\n',' ')
            # print(f"RANK {self.global_rank}: GENERATED: [{chunk}]")

            #dist.barrier()
            #chunk = decoded[1].replace('\n',' ')
            #print(f"RANK {self.global_rank}: GENERATED: [{chunk}]")
            
            # expected 1 generated character:
            # 0: o
            # 1: .

            import sys; sys.exit(0)

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

        print_rank(f"{loss=}")
        import sys; sys.exit(0)

        self.model.backward(loss)
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
