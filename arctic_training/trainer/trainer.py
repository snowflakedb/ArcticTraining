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

import deepspeed.comm as dist
#from deepspeed.sequence.layer import UlyssesAttention
from einops import rearrange
class UlyssesAttentionNew(torch.nn.Module):
    """Re-Implementation of DistributedAttention. This implementation enforces the input shape
    to be standard [s, b, heads, dim_per_head] form. Any deviation from this shape will raise an error
    should be handled directly in the attn passed to the forward. 
    
    The primary reason for the re-implementation is to make this less error prone, and remove what seemed like 
    bugs in scenarios where batch size > 1 and when using different versions of 
    flash attention each of which takes different input shape. Those should be handled by 
    the actual attn implementation, and not by this module.

    Arguments:
        local_seq_length (int): local sequence length per GPU,
        global_seq_length (int): actual sequence length,
        batch_size (int): batch size,
        attn_head_size (int): size of each attention head,
        attn_head_count (int): total number of attention heads,        
        process_group (dist.ProcessGroup): Ulysses Process Group
    """

    def __init__(
        self,
        self_attn,
        local_seq_length: int,
        global_seq_length: int,
        batch_size: int,
        attn_head_count: int,        
        attn_head_size: int,
        process_group: dist.ProcessGroup
    ) -> None:

        super(UlyssesAttentionNew, self).__init__()
        self.self_attn = self_attn
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        
        assert attn_head_count % self.world_size == 0, f"Attention head count {attn_head_count} is not divisible by world size {self.world_size}"
        
        self.local_seq_length = local_seq_length
        self.global_seq_length = global_seq_length
        self.batch_size = batch_size
        
        self.attn_head_size = attn_head_size
        self.attn_head_count = attn_head_count
        
        self.required_input_shape = torch.Size([local_seq_length, \
                                                batch_size, \
                                                attn_head_count, \
                                                attn_head_size])
        
        self.required_context_shape = torch.Size([global_seq_length, \
                                                batch_size, \
                                                attn_head_size * attn_head_count // self.world_size])
        
    def _combine_local_sequences(self, query, key, value) -> Tuple[Tensor, Tensor, Tensor]:
        
        def combine_sequence(input):
            input = input.reshape([self.local_seq_length, \
                                self.batch_size, \
                                self.world_size, \
                                self.attn_head_count // self.world_size, \
                                self.attn_head_size])    
            input = rearrange(input, 's b w ... -> w s b ...').contiguous()
            
            output = _DimZeroAllToAll.apply(self.process_group, input)
        
            output = output.reshape([self.global_seq_length, *output.shape[2:]]).contiguous()
            return output
        
        return combine_sequence(query), combine_sequence(key), combine_sequence(value)
        
    def _partition_global_sequence(self, input) -> Tensor:
        
        input = input.reshape([self.world_size, \
                            self.local_seq_length, \
                            self.batch_size, \
                            self.attn_head_size * self.attn_head_count // self.world_size]).contiguous()    
        
        output = _DimZeroAllToAll.apply(self.process_group, input)
        output = rearrange(output, 'w s b ... -> s b w ...')
            
        #s b w d --> s b wd
        output = output.reshape([*output.shape[:2], -1]).contiguous()
        
        return output
        
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """ forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            ### XXX: attn (Callable): Attention method to call
            args: other args

        Returns:
            * output (Tensor): context output
        """
        

        print(f"{query.shape=}")
        print(f"{key.shape=}")
        print(f"{value.shape=}")
        print(f"{self.required_input_shape=}")  
        assert query.shape == key.shape == value.shape == self.required_input_shape, \
                    f"One of the input tensors does not match the required shape {self.required_input_shape}"
        
        query_layer, key_layer, value_layer = self._combine_local_sequences(query, key, value)
            
        context_layer = self.attn(query_layer, key_layer, value_layer, *args)
        
        assert context_layer.shape == self.required_context_shape, \
                    f"The context shape {context_layer.shape} is not as expected shape {self.required_context_shape}"
        
        output = self._partition_global_sequence(context_layer)
        
        return output

# class UlyssesAttentionHF(UlyssesAttentionNew):
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

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        #         logger.warning_once(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # XXX: 
        attention_interface = ALL_ATTENTION_FUNCTIONS["ulysses"]
        print(f"custom attention on {torch.distributed.get_rank()}")

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

        attn_implementation_real =  transformers.models.llama.modeling_llama.eager_attention_forward

        #from deepspeed.sequence.layer import DistributedAttention
        uattn = UlyssesAttentionNew(
            self_attn=attn_implementation_real,
            local_seq_length=self.config.data.max_length // mpu.get_sequence_parallel_world_size(),
            global_seq_length=self.config.data.max_length, 
            batch_size=self.config.micro_batch_size, 
            attn_head_size=model_config.num_attention_heads, 
            attn_head_count=model_config.hidden_size // model_config.num_attention_heads, 
            process_group = mpu.get_sequence_parallel_group(),
        )

        def uattn_wrapper(
            module: torch.nn.Module,        
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            *args,
            **kwargs,               
        ) -> Tuple[torch.Tensor, None]:
            
            # HF incoming shapes are:
            # [batch_size, num_heads, seqlen, head_size]
            # for MQA key/value num of heads is different from query
            # UA expects [seqlen, batch_size, head_size, num_heads]
            # print(f"{query.shape=}")
            # print(f"{key.shape=}")
            # print(f"{value.shape=}")
            # print(f"{self.required_input_shape=}")   
            query = rearrange(query, 'b h s w -> s b w h')
            key = rearrange(key, 'b h s w -> s b w h')
            value = rearrange(value, 'b h s w -> s b w h')

            attn_output = uattn(
                query=query,
                key=key,
                value=value,
                # XXX: fixme
                #*args,
                #**kwargs
            )
            return attn_output, None           

        ALL_ATTENTION_FUNCTIONS["ulysses"] = uattn_wrapper

        dschf = HfDeepSpeedConfig(self.config.deepspeed)  # noqa: F841
        model_factory = self.config.model.factory(self)
        self.model = model_factory()

        optimizer_factory = self.config.optimizer.factory(self)
        self.optimizer = optimizer_factory()

        scheduler_factory = self.config.scheduler.factory(self)
        self.scheduler = scheduler_factory()

        #import traceback
        #traceback.print_stack(limit=6)

       
        # # seq_length = len(batch["input_ids"][0])
        # # sp_world_size = groups._get_sequence_parallel_world_size()
        # # sp_rank = groups._get_sequence_parallel_rank()
        # # chunk_len = math.ceil(seq_length / sp_world_size)

        # self.config.data.max_length
        # self.config.micro_batch_size
        # self.model.config.num_attention_heads
        # self.model.config.hidden_size 
        # mpu.get_sequence_parallel_world_size()
        # mpu.get_sequence_parallel_group()


        #for n,p in m.named_parameters():
        #  print(name, )

        #print(f"MDDDD  {self.model}") 
        #print(f"MDDDDL  {self.model.model.layers}") 
        # monkeypatch Attention
        # XXX: this is harcoded for LLama - replace with sub-module search matching /Attention/
        # for idx in range(len(self.model.model.layers)):
        #     x = self.model.model.layers[idx].self_attn
        #     self.model.model.layers[idx].self_attn = 

        # for idx in range(len(self.model.model.layers)):
        #     x = self.model.model.layers[idx].self_attn
        #     self.model.model.layers[idx].self_attn = DistributedAttention(
        #         x, mpu.get_sequence_parallel_group())

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
        dist.barrier()
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

        
        batch = {'input_ids': torch.tensor([[    1,  4093,   198, 18588,   354,   297,    29, 23900,  3941,  1015,  1691,   640,   284, 10551, 10983, 20529,
           327,  9599,  4604,    30,     2,   198,     1,   520,  9531,   198,    57,  1326,   982,   457,   260,  4190,
           288,  1235,   253,  4090,  6603,   297,    29, 23900,  3941,   335,   957,  1038,    28,   564,   339,   416,
          1538,   354, 10925,   282,   638,   288,  1235,   354,   297,    29, 23900,  3941,  1015,   574,   640,   284,
         10551,  2290, 22162,   327,  9599,  4604,    30,  3726,   359,   260,  3301,    42,  1116,    33,    30,  5427,
           614,   253,  1421,  1357,   351,  6460,   105,   335, 31408,   284,   574,   640,   411,  1695,   260, 10686,
          6388,   335,   260,   574,   640, 10822,  3013,    30,  1116,    34,    30,  5843,   260,  2914,  8334,   284,
          1464,   260,  2371,  5158,   327,   260,   297,    29, 23900,  2530,    30,   669,  2978,   260,  1478,  3013,
            28,  1406,  5158,    28, 11316,  6657,  3013,    28,   284, 42278,  3013,    30,  1116,    35,    30,  1071,
         15123,   260,   574,   640,  4249,  4840,    28,   527,  3445,  4054,   614,   260,  3024,  3400,    28, 12497,
          2337,    28,  9599,  2337,    28,   284,  4106, 34108,    30,  1116,    36,    30,  5427,   614,   260,  2290,
         22162,  9599, 23147,   411, 10551,   260,  2290, 22162, 14615,   618,   260,   297,    29, 23900,  2530,    30,
           669,   416,   325,  2294,  1015,   260,  2290, 22162,   262,    84,   538,   355,   411,  1015,   574,   640,
           506,  2837,    29,   254,  7657,   351,  2290, 22162,    30,  1116,    37,    30,  4246,   260,   297,    29,
         23900,  3941,   288,  2381,   338,  3117,   314,  1891,   347,  3393,    30,   669,  2978,  3728,   260, 11316,
          6657,    28, 42278,   980,    28,   284,  9599,  4604,    30,  1116,    38,    30,  1413,  1617,   260,   297,
            29, 23900,  3941,   288,   253,  2262,  1357,    28,  2267,   335,   253,  3877, 17042,  2754,   355,   253,
          6249,  3941,   702,  1243,    99,   355, 23833,  6249,    30,  1116,    39,    30, 17897,   260,   297,    29,
         23900,  3941,   288,  2381,   338,   357,   314,  4108, 12250,   284, 10575,   750,  1974,   338,  8120,    30,
           198,   198,  1717,  1695,   623,  3301,    28,   346,   416,  1235,   354,   297,    29, 23900,  3941,  1015,
           574,   640,   284, 12131,  2290, 22162,   327,  9599,  4604,    30,  1929,   451,  3941,    28,   346,   416,
          2626,  6348,   253,  5253,   284, 11541,   970,   288,  6434,  2329,   284,  2290,   327,   480, 17744,    30,
             2,   198,     1,  4093,   198,  1589,   506,   253,  1123,  4335,  1225,    28,   564,   339,  6737,   702,
           288,   699,   540,   563, 10551, 10983, 20529,   351,  1691,   640,    30,  1978,   346,  1538,   540,  5861,
          1096,   335,   260,  3301,  2773,   281,  4054,   614, 10983, 20529,   347,   253,  9599, 23147,    47,     2,
           198,     1,   520,  9531,   198, 34355,    28,  1535,   359,   540,  5861,  3301,  2773,   281, 10551, 10983,
         20529,   351,  1691,   640,    42,  1116,    33,    30,  7538,   614,   327,   253, 10983, 20529,  2051,   355,
          1993,   281,   288,   469,  3832,  2051,    30,   216,    34,    30,  2351,   288,   260, 10983, 20529, 46324,
         39924,  4265,   284,  1464,   253, 38804, 12077,   566,    30,   216,    35,    30,  5722, 28720,   288,   260,
           566,  4840,   284,  5803,   335,   260,   476, 25857,    18, 10147,   288,   820,   260,  5688,  8927,   284,
          4911,    30,   216,    36,    30,  7779,   253,   725,  9599,  1341,   281,  1691,   640,   327, 10983, 20529,
           411,  2045,   288,   260,   476, 25017,   358, 14890,    18,  3246,   282,   260,  1691,   640, 11931,  8253,
            30,   216,    37,    30, 10452,   476,  8103,   640,  9533, 25017,   358, 15813,  9533, 25017, 20529,  2516,
          7041,    18,   347,   260,  8340,    30,   216,    38,    30, 10760,   469, 10983, 20529,  5688,  8927,   284,
          4911,   281,   260,   476, 29039,  3649,    18,  3246,   282,   260,  9599,  1341,  4840,    30,   216,    39,
            30,  4363,  1282,   750,   550,  9696,  3416,   346,  1277,   327,   260,  9599,  1341,    28,   715,   347,
         13043, 12235,   355,  1686,  3559,   266,    30,   216,    40,    30, 45151, 10983, 20529, 16405,  7023,   447,
           281,  1691,   640,   411,  2045,   288,   260,   476, 10280,   447, 22806,    18,  3246,   282,   260,  1691,
           640, 11931,  8253,    30,   216,    41,    30, 10375,   335,   260,   476, 25017,   358, 14890,    18, 10147,
           284,  2545,   260,   476, 25017, 20529, 16405,    18,  3985,    30,   216,    33,    32,    30,  4246,   260,
         10983, 20529,  7657,   411,  4990,   253,  1406,   288,   260,  6657,    28, 21983,   288,   260, 42278,    28,
           284, 10019, 10983, 20529,   347,   260,  9599,  1341,    30,   216,    33,    33,    30, 18254,   253,  1028,
         13043,   281, 10983, 20529,   288,  2381,   338,   260,  9599,   314,  8484,  5000,    30,  1848,   506,   253,
           904,    29,  4638,  9912,   282,   638,   288, 12131, 10983, 20529,   351,  1691,   640,    30,  2015,   282,
           260,  3841,  2773,   281,  4054,   614, 10983, 20529,   347,   253,  9599, 23147,   523,  3749,   335,   469,
          1678,   722,  1671,   284,  1861,  4292,    28,   564,   451,   868,  1928,   346,   253,  1123,  4335,  1225,
           327,  2967,  2841,   351,  1691,   640,   284, 10983, 20529,  7657,    30,     2,   198,     1,  4093,   198,
          7306,   346,   597,  1538,   540,  1096,   335,   638,   288,   932,   614, 12497,  2337,   281,  1691,   640,
            47,   339,  5248,   441,  2090,   837,   288,  1120,    30,     2,   198,     1,   520,  9531,   198, 34355,
            28,  1535,   359,   540,  5861,  3301,  2773,   281,  4054,   614, 12497,  2337,   281,  1691,   640,    42,
          1116,    33,    30,  2351,   288,   260,   476,  4370,  6487, 14890,    18,  3246,   282,   260,  1691,   640,
         11931,  8253,    30,   216,    34,    30, 10375,   335,   260,   476,  5529, 46757,  7951,    18,  7885,   288,
          1464,   253,   725, 12497,  1341,    30,   216,    35,    30, 10760,   253,  1462,   327,   260, 12497,  1341,
            28,   715,   347,   476, 23013, 46757,    18,   355,   476,  2516,  7041, 46757,  2227,   216,    36,    30,
         10452,   253, 17993,   288,  3346,   260, 12497,  2421,  1552,   335,   260,  1686,  2613,    28,  1686,  2719,
            28,   355,   550,  2212,    30,  1691,   640,  2216,   351,  1545,  2837,    29,   254, 38401,    28,   355,
           346,   416,  1464,   469,  1038,  2929, 17993,    30,   216,    37,    30,  4363,  1282,   260, 12497,  9834,
           837,   260, 12497,  1341,   523,   325,  1770,    30, 46757,  9834,   416,   325,  1552,   335,  1798,    28,
          1215,    28,   355, 33014,  2909,    30,   216,    38,    30, 10452,   253, 12497,  6881,   327,   260, 12497,
          1341,    30, 46757,  6585,   359,   804,   288,  1528,  2320,   351,  1887, 12497,  4292,    30,   216,    39,
            30,  4363,  1282,   750,  9416,   355,  7250,   335,   260, 12497,  1341,    28,   715,   347,  5869,  1686,
          1685,   355,  5428,  2613,    30,   216,    40,    30, 16326,   260, 12497,  1341,   288,   803,   357,   288,
           469,   297,    29, 23900,  2530,    30,   216,    41,    30,  4246,   260, 12497,  1341,   411,  4990,   253,
          1406,   288,   260,  6657,   284, 21983,   288,   260, 42278,    30,   378, 12497,  1708,   868,   325,  8449,
          1552,   335,   260, 12497,  1341,   346,   932,   614,    30,  1848,   506,   253,   904,    29,  4638,  9912,
           282,   638,   288,   932,   614, 12497,  2337,   281,  1691,   640,    30,  2015,   282,   260,  3841,  2773,
           281,  4054,   614, 12497,  2337,   523,  3749,   335,   469,  1678,   722,  1671,   284,  1861,  4292,    28,
           564,   451,   868,  1928,   346,   253,  1123,  4335,  1225,   327,  2967,  2841,   351, 12497,   281,  1691,
           640,    30,     2,   198,     1,  4093,   198, 16937,   327,   260,  5861,  1096,   335,  4054,   614,   260,
           297,    29, 23900,  3941,   351,   574,   640,   284, 10551,  2290, 22162,    30,  1978,   346,  1538,   540,
          1096,   335,   638,   288,   932,   614,  4106, 34108,   281,   574,   640,    47,   339,  1277,   288,   919,
          2090,   957,  6348,  3796, 34108,   563,   480,  9112,   284, 12497,  3559,    30,     2,   198,     1,   520,
          9531,   198, 47302,    17,  3726,   506,   638,   346,   416,   932,   614,  4106, 34108,   281,  1691,   640,
            42,  1116,    33,    30,  2351,   288,   260,   476, 47665,  7951,    18,  3246,   282,   260,  1691,   640,
         11931,  8253,    30,  1116,    34,    30, 10452,   253, 11841,  1341,   429,   260,  1770,  3416,    30,  1691,
           640,  6569,  1545, 11841,  2337,    28,   715,   347, 15620,  7148,    28, 25054, 25665,    28, 23767, 23544,
            28,   284,   540,    30,  1116,    35,    30, 10760,   260,  2371,  4840,   327,   469, 11841,  1341,    30,
          1068,  1183,    28,   585,   346,  3525, 15620,  7148,    28,   346,  3060,   737,   288,  4137,   469, 15620,
          7148,  6064,  2014,    28,  2399,    28, 13910,    28,   284,  8824,    30,  1094,   346,  3525, 25054, 25665,
           355, 23767, 23544,    28,   346,  3060,   737,   288,  4137,   469, 12077,  1646,    30,  1116,    36,    30,
         45151,  4106, 34108,   327,  1461,  2466,    28,   715,   347,   725,  9112,    28,  1686, 11029,    28, 24710,
           491,    28,   284, 42149,    30,  1206,   416, 25718,   260,  4106, 20956,   338,   359,  2362,   327,   971,
          2121,    30,  1116,    37,    30, 16326,   469,  4840,   288,  5202,  4106, 34108,    30,  1116,    38,    30,
          4246,   469,  4106, 34108,   411, 10482,   354,  1686,   284, 11160,   338,   346,  3796,   260,  4106, 20049,
            30,   198,   198,  5195,   506,   253,   904,    29,  4638,  9912,   282,   638,   288,   932,   614,  4106,
         34108,   281,  1691,   640,    30,  2015,   282,   260,  3841,  2773,   281,  4054,   614,  4106, 34108,   523,
          3749,   335,   260,  1678, 11841,  1341,   284,  4106, 20956,   346,  3525,   288,   722,    30,  1423,    28,
           451,   868,  1928,   346,   253,  1123,  4335,  1225,   327,  2967,  2841,   351,  4106, 34108,   281,  1691,
           640,    30,     2,   198,     1,  4093,   198,  2020,   986,   536,   339,   737,   288, 27290,   260,  9973,
           281,  3380,   913,  1092,  6017,   601,    47, 19842,  2289,   288,    42,   216,    33,    30, 10818,  9973,
           284, 25579,   281,  4011,   913,  2268,   355,  6757,  1793,   913,  7313,  2437,    30, 18565,  9973,   351,
          3380,   913,   284, 27290, 15115,    30,   198,    34,    30,  9373,  3380,   913,   365,    36, 12382,    25,
           281,   253,  1507,  1267,   338,   553,   253,  9030,    29, 22866, 19033,    30,  5419,   273, 17059,  9973,
            28, 12917,   539,   703,    28, 22031,    30,   198,    35,    30, 20512,   288,   253, 17663,   284,  2208,
          2817,   288,  5641,    28,  4825,   357,   418,   253, 30398,    30, 13166,    28,  4993,    28,   327,   216,
            34,   355,   216,    35,  2737,   355,  1793, 14533,    30,  1094,   540,   913,   314,  2350,  1811,   803,
         16891,   913,    28,   253,  7118,   355,   588,   418,   253,   655,    30,   198,    36,    30,  5419,  5420,
           990,  9973,   359,  2375,    30,   198,    37,    30, 32846,   618, 12505,   284,  3809,   351,   913,   347,
         14562,    30,  3315,   441,  3809,   260,  2444, 22031,   355, 12917,   539,   703,    30,   198,    38,    30,
          8992,   351, 18346,   288,  5350,   365,    33, 36850,    25,   567, 12505,   282,  9973,    31, 11699,   373,
            30,     2,   198,     1,   520,  9531,   198,  5449,   288,   260,  6388,  1836,    28,   260,  9973,   868,
           325, 31925, 15115,   281,  3380,   913,    30,     2,   198,     1,  4093,   198,  7306,   346,  5007,   549,
          1163,   638,  1083,  5420,   288,   803,   288,   260,  9973,   990,   502,   359,  2375,    47,     2,   198,
             1,   520,  9531,   198,   504,  6388,   536,   441, 13265,   260,  1902,   282,  5420,   288,   803,   288,
           260,  9973,   990,   502,   359,  2375,    30,  1206,   416,   803,  5420,   288,  6309,    28,  4335,   351,
           253,  1165,  1902,   284,  7443,  4990,   540,   347,  2350,    30,     2,   198,     1,  4093,   198, 39122,
            28,  3363,   357,    30,  1978,   346,  5007,   549,  1163,   638,   986,   339,   868,  4857,   260,  9973,
           327,   990, 32808,   601, 15115,    47,     2,   198,     1,   520,  9531,   198,  5449,   288,   260,  6388,
          1836,    28,   260, 31925,  9973,   868,   325, 11410,   327,   216,    34,   355,   216,    35,  2737,    28,
           355,  1793, 14533,    30,     2,   198,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2],
        [    1,  4093,   198, 19161,   354,  1158,    29,   277,   563,   260,  1645,   282,  2412,  1363,   335,   260,
          8252,   282,  3961,  5574,   284,  5136,   281,  1642,    29, 11795,  2429,    30,     2,   198,     1,   520,
          9531,   198, 20799,  1363,   314,   253,  2224,  7613,   338,  5547,   897,  2325,   282,   653, 22355,    30,
          1963,   282,   260,   768,  1546,  5029,   282,  2412,  1363,   314,   260,  8252,   282,  3961,  5574,   284,
          5136,   281,  1642,    29, 11795,  2429,    30,  3959,   288,   253,  1378,   411,   260,  6383, 11799,  6404,
            28,  2412,  1363,   314,  7053,   260,  2174,   284,  9044,   282,  5574,   284,  5136,    30,   378,  1378,
          3935,   335,   288,  1215,   338,   701,  2242,   281,  1642,    29, 11795,  2429,   359,   768,  3900,   411,
           451,  7613,    30,   198,   198, 48529,   284, 10103,  2262,   314,  7100, 13840,   411,  2412,  1363,    30,
         12399,   352,  4018,  4764,   284,   540,  7722, 23913,   919,   357,  1990,   327,  5283,   288,  1075,  6020,
          6026,    30, 31928,  3947,   715,   347, 21834,    28, 31054,    28,   284, 15515,   597,  3055,  6020,    28,
          2899,   288, 20659,   284,  2061,  5770,    30,  7840,  1363,   553,   597,  2842,   288,  1971,   281,   260,
         11706,   282,  9137,   284, 15354, 12064,    28,   527,  5547,   260,  8252,   282,  3961,  2690,    30,   198,
           198, 19229,    29, 11795,  2429,   359,  2755,  6876,   288,   260,  2165,   282,  2412,  1363,    30,  1216,
          2429,  1129,  6654,   335,  1165,   284,  5641,  3964,  5283,   617,   536,   441,   457,   260,  1952,   288,
          2930,   288,   260,  4340,  3947,  1920,    30,   378,   904,  1708,   282,  1835,   284,   550,  9577,  2309,
           288, 12676,   260,  2165,   282,  2412,  1363,   314,   597,   253,  1739, 10269,   327,  1165,  5283,    30,
           669,  1530,   338,  1642,    29, 11795,  2429,   359,   540,  2003,   288,  2715,  1114, 23406,   284, 21999,
           347,   253,   966,   282,  2412,  1363,    30,   198,   198, 15235, 23406,   314,   253,  2969,  1732,   281,
           260,  1797,  1918,    28,   284,  2412,  1363,   523,   805, 23943,   451,  2843,    30,  2904,   553,  3057,
           338,  1642,    29, 11795,  2429,   359,   540,  2003,   288,  6652,   429,  2714,    29,  4235,  3369,   715,
           347,  7671,    28,  4348,    28,   284,  1911,  1611,    30, 20672,   282,  1594,   288,  3961,  5574,   284,
          5136,   553,   719,  4329,   347,   582,   282,   260,  1739, 18990,   288,  3183,   864,  4530,  1486,  1642,
            29, 11795,  3168,    30,  1032,  2412,  1363,  4851,   288,  2151,  6429,  2262,    28,   260,  8252,   282,
         10994,  3961,  5574,   284,  5136,   523,  1438,   908,   540,  3435,    28, 34044,   864, 12570,   281,  6876,
          2429,    30,   198,   198,  1882,  1251,  1188,  2353,   288,  2369,   260,  1645,   282,  2412,  1363,   335,
           260,  8252,   282,  3961,  5574,   284,  5136,   281,  1642,    29, 11795,  2429,    30,  1963,   970,   288,
           536,   451,   314,   288,  1199,  1165,   284,  5641,  3964,  5283,   617,   359,   768,  3900,   411,  2412,
          1363,    30, 43245,   281,  2412,    29, 24914,  5995,  2718,   715,   347,  6429, 30954,    28,  2763,  4949,
            28,   284,   540,  3758, 11034,  2337,   416,   724,  5283,  2930,   288,   260,  4340,  2412,    30,   378,
          1693,   416,   597,  1538, 19768,   284, 14313,   288,   724,  1165,  5283,  1594,  1835,   284,   550,  9577,
          2309,   327,  2412,    29, 24914,  2718,    30,   198,   198,   788,  4446,    28,  2412,  1363,   314,   253,
          1546,  2896,   288,   260,  8252,   282,  3961,  5574,   284,  5136,   281,  1642,    29, 11795,  2429,    30,
          1046,  1251, 18537,   260,  1923,   282,  6876,  2429,   411,  2498,  2353,   288,  2369,   260,  1645,   282,
          2412,  1363,   335,  5995,    30, 28411,  1165,    29,  5904,  5283,   284, 11650,   281,  2412,    29, 24914,
          2718,   359,  3202,  3301,   288,  2381,   338,  2573,   553,  1594,   288,  2458,   284, 15627,  3111,    28,
          7142,   282,   480,  4546,    30,     2,   198,     1,  4093,   198,  1348,   314,   253,  1109,  1120,    28,
           564,   416,   346,  1928,   549,   540,  3480,   282,   638,  2412,  1363,  5547,   260,  8252,   282,  3961,
          5574,   284,  5136,   281,  1642,    29, 11795,  2429,    47,     2,   198,     1,   520,  9531,   198, 47302,
            17,  3726,   359,   634,  3171,  3480,    42,  1116,    33,    30, 47210,  6429, 11783,    42,  1963,   282,
           260,   768,  1546,  5029,   282,  2412,  1363,   335,  5995,   314,  8422,  6429, 11783,    30,   669,   314,
          2755,  2629,   327,  5574,   284,  5136,    28,   527,   359,  5857,   288,  1971,   281,  2779,   284,   913,
          8252,    30,  1550,  6429, 11783,  6302,    28,   260,  4319,   282,  3961,  2690, 10118,   288,   685,   614,
            28,  1625,   357,  1181, 10994,   327,  1642,    29, 11795,  2429,    30,  1116,    34,    30,  4643, 20701,
            42,  7840,  1363,   314,  4439,   540,  7722,   284,  3523, 23913,   281,   800,  4286,    28,   527,  2022,
           357,  1990,   327,  5283,   288,  9475,   368,   480,  6020,    30,   669,   416,  1022,   288,  3954, 11783,
           284,  2208,    29,  7535,  2690,    30,   533,   634,  2199,    28,  5283,   654,   325,  5657,   288,  8039,
           480,  6020, 13587,  1568,   288,  3096,   282,   913,    28,  2030,  4345,   260,  8252,   282,  3961,  5574,
           284,  5136,    30,  1116,    35,    30, 18308, 11678, 31858,    42,  2287,   763,  4764,   284,  4340, 11228,
          3077,   416,   597,  1728,   354,  2081,   281, 11678, 31858,    28,   527,   416,  3055,  6020,   284,  2369,
         11783,    30,   669,   416,   919,   357,  8643,   327,  5283,   288,  2690,  2001,  3961,  2690,   288,  2220,
          3516,    28,   527,   416,  1022,   288, 20659,   284,  2061,  5770,    30,  1116,    36,    30, 16691,   281,
          2969,  1920,    42,  7840,  1363,   314,   597,  4439,  1971,   281,   260, 11706,   284,  3695,   282,  2969,
         12064,    30,   669,   416,  2151,   260,  8252,   282,  1647,  5574,   284,  5136,   281,  1642,    29, 11795,
          2429,    28,  2117,   585,   623,  2429,  6654,   335,   253,  6260,  3175,   282,  6020,   338,   359,  5857,
           288,  1971,   281,  2969,  1920,    30,   198,   198, 20832,    28,  2412,  1363,   314,  1953,   253,  1546,
          1645,   335,   260,  8252,   282,  3961,  5574,   284,  5136,   281,  1642,    29, 11795,  2429,    30,  1428,
          2498,  2353,   288,  2369,   653,  7499,  2811,  4562,   284,  1199,  2412,    29, 24914,  5995,  2718,    28,
           392,   416,   724,  2381,   338,  2573,   553,  1594,   288,  2458,   284, 15627,  3111,    28,  7142,   282,
           480,  4546,    30,     2,   198,     1,  4093,   198, 15635,    17,  1978,   346,   803,   634,  1096,   335,
           638,  2412,  1363,  5547,   260,  6367,   284,  4759,   282,  3961,  2690,   281,  1642,    29, 11795,  2429,
            47,     2,   198,     1,   520,  9531,   198, 42686,  8234,    17,  7840,  1363,   597,  5547,   260,  6367,
           284,  4759,   282,  3961,  2690,    28,   527,   416,   457,   253,  1546,  1645,   335,   624,  8252,   284,
         29497,   281,  1642,    29, 11795,  2429,    30,  3726,   359,   253,  1443,  1853,   281,   527,   451,   416,
          4317,    42,  1116,    33,    30, 18607, 23551,    42, 31928,  3947,  2466,   702, 15515,    28, 21834,    28,
           284, 24670,   416,  3055,  6367,  5730,   702,  8364,    28, 12223,    28,   284, 14796,    30,   669,   416,
           919,   357,  1990,   327,  3961,  2690,   288,   325, 15326,   429,  2969,  4286,   288,  1642,    29, 11795,
          2429,    30,   533,  1706,    28,  3977,  5770,   654,  2081,  1568,   288,  6367, 23551,    28,  2899,   288,
          2061,  5770,   327,  6461,    30,   216,    34,    30, 14576,   282,  4759,  5834,    42, 31928,  3947,  2466,
           416,   597,  3055, 45185,  4759,  5834,    28,   527,   359,  1895,   327,  8314,  3961,  2690,    30,  1550,
           623,  5834,   359,  3590,    28,   357,   416,   325,   540,  1990,   288,  3317,   284,  4249,  3961,  2690,
            28,  2899,   288, 48595,   284,  3368,    30,   669,   416,   919,  3961,  5574,   284,  5136,   540,  6280,
           284,  1181,  1770,   281,  1642,    29, 11795,  2429,    30,   216,    35,    30, 18308,  1114,  3368,    42,
          7840,  1363,   416,   597,  1022,   288,  2420,  1114,  3368,    28,   527,   553,   253,  1739,  1645,   335,
           260,  8252,   284, 29497,   282,  3961,  2690,    30, 31928,  3947,  2466,   416,  3055,  6020,    28,  4439,
           601,   288,   685,   288,  3368,    30,   533,  1706,    28,   645,  3961,  2690,   314,   441,  5000,  6314,
            28,   357,   416, 28255,   284,   685,   288,  3368,    30,   669,   416,  1022,   288, 20659,   284,  2061,
          5770,   327,  6461,    28,  2117,   281,  1642,    29, 11795,  2429,   837,  1594,   288,  3961,  2690,   314,
          2221,  3435,    30, 18813,    28,  2412,  1363,  5547,   260,  6367,   284,  4759,   282,  3961,  2690,   281,
          1545,  1853,    28,   527,   416,   457,   253,  1739,  1645,   335,   624,  8252,   284, 29497,   281,  1642,
            29, 11795,  2429,    30,  1428,  2498,  2353,   288,  2369,   653,  7499,  2811,  4562,   284,  1748,   281,
          2412,    29,   375, 13727,  5730,    28,   392,   416,   724,  2381,   338,  2573,   553,  1594,   288,  2458,
           284, 15627,  3111,    28,  7142,   282,   480,  4546,    30,     2,   198,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
             2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2,     2]]), 
             'labels': torch.tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,    57,  1326,   982,   457,   260,  4190,
           288,  1235,   253,  4090,  6603,   297,    29, 23900,  3941,   335,   957,  1038,    28,   564,   339,   416,
          1538,   354, 10925,   282,   638,   288,  1235,   354,   297,    29, 23900,  3941,  1015,   574,   640,   284,
         10551,  2290, 22162,   327,  9599,  4604,    30,  3726,   359,   260,  3301,    42,  1116,    33,    30,  5427,
           614,   253,  1421,  1357,   351,  6460,   105,   335, 31408,   284,   574,   640,   411,  1695,   260, 10686,
          6388,   335,   260,   574,   640, 10822,  3013,    30,  1116,    34,    30,  5843,   260,  2914,  8334,   284,
          1464,   260,  2371,  5158,   327,   260,   297,    29, 23900,  2530,    30,   669,  2978,   260,  1478,  3013,
            28,  1406,  5158,    28, 11316,  6657,  3013,    28,   284, 42278,  3013,    30,  1116,    35,    30,  1071,
         15123,   260,   574,   640,  4249,  4840,    28,   527,  3445,  4054,   614,   260,  3024,  3400,    28, 12497,
          2337,    28,  9599,  2337,    28,   284,  4106, 34108,    30,  1116,    36,    30,  5427,   614,   260,  2290,
         22162,  9599, 23147,   411, 10551,   260,  2290, 22162, 14615,   618,   260,   297,    29, 23900,  2530,    30,
           669,   416,   325,  2294,  1015,   260,  2290, 22162,   262,    84,   538,   355,   411,  1015,   574,   640,
           506,  2837,    29,   254,  7657,   351,  2290, 22162,    30,  1116,    37,    30,  4246,   260,   297,    29,
         23900,  3941,   288,  2381,   338,  3117,   314,  1891,   347,  3393,    30,   669,  2978,  3728,   260, 11316,
          6657,    28, 42278,   980,    28,   284,  9599,  4604,    30,  1116,    38,    30,  1413,  1617,   260,   297,
            29, 23900,  3941,   288,   253,  2262,  1357,    28,  2267,   335,   253,  3877, 17042,  2754,   355,   253,
          6249,  3941,   702,  1243,    99,   355, 23833,  6249,    30,  1116,    39,    30, 17897,   260,   297,    29,
         23900,  3941,   288,  2381,   338,   357,   314,  4108, 12250,   284, 10575,   750,  1974,   338,  8120,    30,
           198,   198,  1717,  1695,   623,  3301,    28,   346,   416,  1235,   354,   297,    29, 23900,  3941,  1015,
           574,   640,   284, 12131,  2290, 22162,   327,  9599,  4604,    30,  1929,   451,  3941,    28,   346,   416,
          2626,  6348,   253,  5253,   284, 11541,   970,   288,  6434,  2329,   284,  2290,   327,   480, 17744,    30,
             2,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100, 34355,    28,  1535,   359,   540,  5861,  3301,  2773,   281, 10551, 10983,
         20529,   351,  1691,   640,    42,  1116,    33,    30,  7538,   614,   327,   253, 10983, 20529,  2051,   355,
          1993,   281,   288,   469,  3832,  2051,    30,   216,    34,    30,  2351,   288,   260, 10983, 20529, 46324,
         39924,  4265,   284,  1464,   253, 38804, 12077,   566,    30,   216,    35,    30,  5722, 28720,   288,   260,
           566,  4840,   284,  5803,   335,   260,   476, 25857,    18, 10147,   288,   820,   260,  5688,  8927,   284,
          4911,    30,   216,    36,    30,  7779,   253,   725,  9599,  1341,   281,  1691,   640,   327, 10983, 20529,
           411,  2045,   288,   260,   476, 25017,   358, 14890,    18,  3246,   282,   260,  1691,   640, 11931,  8253,
            30,   216,    37,    30, 10452,   476,  8103,   640,  9533, 25017,   358, 15813,  9533, 25017, 20529,  2516,
          7041,    18,   347,   260,  8340,    30,   216,    38,    30, 10760,   469, 10983, 20529,  5688,  8927,   284,
          4911,   281,   260,   476, 29039,  3649,    18,  3246,   282,   260,  9599,  1341,  4840,    30,   216,    39,
            30,  4363,  1282,   750,   550,  9696,  3416,   346,  1277,   327,   260,  9599,  1341,    28,   715,   347,
         13043, 12235,   355,  1686,  3559,   266,    30,   216,    40,    30, 45151, 10983, 20529, 16405,  7023,   447,
           281,  1691,   640,   411,  2045,   288,   260,   476, 10280,   447, 22806,    18,  3246,   282,   260,  1691,
           640, 11931,  8253,    30,   216,    41,    30, 10375,   335,   260,   476, 25017,   358, 14890,    18, 10147,
           284,  2545,   260,   476, 25017, 20529, 16405,    18,  3985,    30,   216,    33,    32,    30,  4246,   260,
         10983, 20529,  7657,   411,  4990,   253,  1406,   288,   260,  6657,    28, 21983,   288,   260, 42278,    28,
           284, 10019, 10983, 20529,   347,   260,  9599,  1341,    30,   216,    33,    33,    30, 18254,   253,  1028,
         13043,   281, 10983, 20529,   288,  2381,   338,   260,  9599,   314,  8484,  5000,    30,  1848,   506,   253,
           904,    29,  4638,  9912,   282,   638,   288, 12131, 10983, 20529,   351,  1691,   640,    30,  2015,   282,
           260,  3841,  2773,   281,  4054,   614, 10983, 20529,   347,   253,  9599, 23147,   523,  3749,   335,   469,
          1678,   722,  1671,   284,  1861,  4292,    28,   564,   451,   868,  1928,   346,   253,  1123,  4335,  1225,
           327,  2967,  2841,   351,  1691,   640,   284, 10983, 20529,  7657,    30,     2,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 34355,
            28,  1535,   359,   540,  5861,  3301,  2773,   281,  4054,   614, 12497,  2337,   281,  1691,   640,    42,
          1116,    33,    30,  2351,   288,   260,   476,  4370,  6487, 14890,    18,  3246,   282,   260,  1691,   640,
         11931,  8253,    30,   216,    34,    30, 10375,   335,   260,   476,  5529, 46757,  7951,    18,  7885,   288,
          1464,   253,   725, 12497,  1341,    30,   216,    35,    30, 10760,   253,  1462,   327,   260, 12497,  1341,
            28,   715,   347,   476, 23013, 46757,    18,   355,   476,  2516,  7041, 46757,  2227,   216,    36,    30,
         10452,   253, 17993,   288,  3346,   260, 12497,  2421,  1552,   335,   260,  1686,  2613,    28,  1686,  2719,
            28,   355,   550,  2212,    30,  1691,   640,  2216,   351,  1545,  2837,    29,   254, 38401,    28,   355,
           346,   416,  1464,   469,  1038,  2929, 17993,    30,   216,    37,    30,  4363,  1282,   260, 12497,  9834,
           837,   260, 12497,  1341,   523,   325,  1770,    30, 46757,  9834,   416,   325,  1552,   335,  1798,    28,
          1215,    28,   355, 33014,  2909,    30,   216,    38,    30, 10452,   253, 12497,  6881,   327,   260, 12497,
          1341,    30, 46757,  6585,   359,   804,   288,  1528,  2320,   351,  1887, 12497,  4292,    30,   216,    39,
            30,  4363,  1282,   750,  9416,   355,  7250,   335,   260, 12497,  1341,    28,   715,   347,  5869,  1686,
          1685,   355,  5428,  2613,    30,   216,    40,    30, 16326,   260, 12497,  1341,   288,   803,   357,   288,
           469,   297,    29, 23900,  2530,    30,   216,    41,    30,  4246,   260, 12497,  1341,   411,  4990,   253,
          1406,   288,   260,  6657,   284, 21983,   288,   260, 42278,    30,   378, 12497,  1708,   868,   325,  8449,
          1552,   335,   260, 12497,  1341,   346,   932,   614,    30,  1848,   506,   253,   904,    29,  4638,  9912,
           282,   638,   288,   932,   614, 12497,  2337,   281,  1691,   640,    30,  2015,   282,   260,  3841,  2773,
           281,  4054,   614, 12497,  2337,   523,  3749,   335,   469,  1678,   722,  1671,   284,  1861,  4292,    28,
           564,   451,   868,  1928,   346,   253,  1123,  4335,  1225,   327,  2967,  2841,   351, 12497,   281,  1691,
           640,    30,     2,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100, 47302,    17,  3726,   506,   638,   346,   416,   932,   614,  4106, 34108,   281,  1691,   640,
            42,  1116,    33,    30,  2351,   288,   260,   476, 47665,  7951,    18,  3246,   282,   260,  1691,   640,
         11931,  8253,    30,  1116,    34,    30, 10452,   253, 11841,  1341,   429,   260,  1770,  3416,    30,  1691,
           640,  6569,  1545, 11841,  2337,    28,   715,   347, 15620,  7148,    28, 25054, 25665,    28, 23767, 23544,
            28,   284,   540,    30,  1116,    35,    30, 10760,   260,  2371,  4840,   327,   469, 11841,  1341,    30,
          1068,  1183,    28,   585,   346,  3525, 15620,  7148,    28,   346,  3060,   737,   288,  4137,   469, 15620,
          7148,  6064,  2014,    28,  2399,    28, 13910,    28,   284,  8824,    30,  1094,   346,  3525, 25054, 25665,
           355, 23767, 23544,    28,   346,  3060,   737,   288,  4137,   469, 12077,  1646,    30,  1116,    36,    30,
         45151,  4106, 34108,   327,  1461,  2466,    28,   715,   347,   725,  9112,    28,  1686, 11029,    28, 24710,
           491,    28,   284, 42149,    30,  1206,   416, 25718,   260,  4106, 20956,   338,   359,  2362,   327,   971,
          2121,    30,  1116,    37,    30, 16326,   469,  4840,   288,  5202,  4106, 34108,    30,  1116,    38,    30,
          4246,   469,  4106, 34108,   411, 10482,   354,  1686,   284, 11160,   338,   346,  3796,   260,  4106, 20049,
            30,   198,   198,  5195,   506,   253,   904,    29,  4638,  9912,   282,   638,   288,   932,   614,  4106,
         34108,   281,  1691,   640,    30,  2015,   282,   260,  3841,  2773,   281,  4054,   614,  4106, 34108,   523,
          3749,   335,   260,  1678, 11841,  1341,   284,  4106, 20956,   346,  3525,   288,   722,    30,  1423,    28,
           451,   868,  1928,   346,   253,  1123,  4335,  1225,   327,  2967,  2841,   351,  4106, 34108,   281,  1691,
           640,    30,     2,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
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
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  5449,   288,   260,  6388,  1836,    28,   260,  9973,   868,
           325, 31925, 15115,   281,  3380,   913,    30,     2,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,   504,  6388,   536,   441, 13265,   260,  1902,   282,  5420,   288,   803,   288,
           260,  9973,   990,   502,   359,  2375,    30,  1206,   416,   803,  5420,   288,  6309,    28,  4335,   351,
           253,  1165,  1902,   284,  7443,  4990,   540,   347,  2350,    30,     2,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  5449,   288,   260,  6388,
          1836,    28,   260, 31925,  9973,   868,   325, 11410,   327,   216,    34,   355,   216,    35,  2737,    28,
           355,  1793, 14533,    30,     2,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100],
        [ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100, 20799,  1363,   314,   253,  2224,  7613,   338,  5547,   897,  2325,   282,   653, 22355,    30,
          1963,   282,   260,   768,  1546,  5029,   282,  2412,  1363,   314,   260,  8252,   282,  3961,  5574,   284,
          5136,   281,  1642,    29, 11795,  2429,    30,  3959,   288,   253,  1378,   411,   260,  6383, 11799,  6404,
            28,  2412,  1363,   314,  7053,   260,  2174,   284,  9044,   282,  5574,   284,  5136,    30,   378,  1378,
          3935,   335,   288,  1215,   338,   701,  2242,   281,  1642,    29, 11795,  2429,   359,   768,  3900,   411,
           451,  7613,    30,   198,   198, 48529,   284, 10103,  2262,   314,  7100, 13840,   411,  2412,  1363,    30,
         12399,   352,  4018,  4764,   284,   540,  7722, 23913,   919,   357,  1990,   327,  5283,   288,  1075,  6020,
          6026,    30, 31928,  3947,   715,   347, 21834,    28, 31054,    28,   284, 15515,   597,  3055,  6020,    28,
          2899,   288, 20659,   284,  2061,  5770,    30,  7840,  1363,   553,   597,  2842,   288,  1971,   281,   260,
         11706,   282,  9137,   284, 15354, 12064,    28,   527,  5547,   260,  8252,   282,  3961,  2690,    30,   198,
           198, 19229,    29, 11795,  2429,   359,  2755,  6876,   288,   260,  2165,   282,  2412,  1363,    30,  1216,
          2429,  1129,  6654,   335,  1165,   284,  5641,  3964,  5283,   617,   536,   441,   457,   260,  1952,   288,
          2930,   288,   260,  4340,  3947,  1920,    30,   378,   904,  1708,   282,  1835,   284,   550,  9577,  2309,
           288, 12676,   260,  2165,   282,  2412,  1363,   314,   597,   253,  1739, 10269,   327,  1165,  5283,    30,
           669,  1530,   338,  1642,    29, 11795,  2429,   359,   540,  2003,   288,  2715,  1114, 23406,   284, 21999,
           347,   253,   966,   282,  2412,  1363,    30,   198,   198, 15235, 23406,   314,   253,  2969,  1732,   281,
           260,  1797,  1918,    28,   284,  2412,  1363,   523,   805, 23943,   451,  2843,    30,  2904,   553,  3057,
           338,  1642,    29, 11795,  2429,   359,   540,  2003,   288,  6652,   429,  2714,    29,  4235,  3369,   715,
           347,  7671,    28,  4348,    28,   284,  1911,  1611,    30, 20672,   282,  1594,   288,  3961,  5574,   284,
          5136,   553,   719,  4329,   347,   582,   282,   260,  1739, 18990,   288,  3183,   864,  4530,  1486,  1642,
            29, 11795,  3168,    30,  1032,  2412,  1363,  4851,   288,  2151,  6429,  2262,    28,   260,  8252,   282,
         10994,  3961,  5574,   284,  5136,   523,  1438,   908,   540,  3435,    28, 34044,   864, 12570,   281,  6876,
          2429,    30,   198,   198,  1882,  1251,  1188,  2353,   288,  2369,   260,  1645,   282,  2412,  1363,   335,
           260,  8252,   282,  3961,  5574,   284,  5136,   281,  1642,    29, 11795,  2429,    30,  1963,   970,   288,
           536,   451,   314,   288,  1199,  1165,   284,  5641,  3964,  5283,   617,   359,   768,  3900,   411,  2412,
          1363,    30, 43245,   281,  2412,    29, 24914,  5995,  2718,   715,   347,  6429, 30954,    28,  2763,  4949,
            28,   284,   540,  3758, 11034,  2337,   416,   724,  5283,  2930,   288,   260,  4340,  2412,    30,   378,
          1693,   416,   597,  1538, 19768,   284, 14313,   288,   724,  1165,  5283,  1594,  1835,   284,   550,  9577,
          2309,   327,  2412,    29, 24914,  2718,    30,   198,   198,   788,  4446,    28,  2412,  1363,   314,   253,
          1546,  2896,   288,   260,  8252,   282,  3961,  5574,   284,  5136,   281,  1642,    29, 11795,  2429,    30,
          1046,  1251, 18537,   260,  1923,   282,  6876,  2429,   411,  2498,  2353,   288,  2369,   260,  1645,   282,
          2412,  1363,   335,  5995,    30, 28411,  1165,    29,  5904,  5283,   284, 11650,   281,  2412,    29, 24914,
          2718,   359,  3202,  3301,   288,  2381,   338,  2573,   553,  1594,   288,  2458,   284, 15627,  3111,    28,
          7142,   282,   480,  4546,    30,     2,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100, 47302,
            17,  3726,   359,   634,  3171,  3480,    42,  1116,    33,    30, 47210,  6429, 11783,    42,  1963,   282,
           260,   768,  1546,  5029,   282,  2412,  1363,   335,  5995,   314,  8422,  6429, 11783,    30,   669,   314,
          2755,  2629,   327,  5574,   284,  5136,    28,   527,   359,  5857,   288,  1971,   281,  2779,   284,   913,
          8252,    30,  1550,  6429, 11783,  6302,    28,   260,  4319,   282,  3961,  2690, 10118,   288,   685,   614,
            28,  1625,   357,  1181, 10994,   327,  1642,    29, 11795,  2429,    30,  1116,    34,    30,  4643, 20701,
            42,  7840,  1363,   314,  4439,   540,  7722,   284,  3523, 23913,   281,   800,  4286,    28,   527,  2022,
           357,  1990,   327,  5283,   288,  9475,   368,   480,  6020,    30,   669,   416,  1022,   288,  3954, 11783,
           284,  2208,    29,  7535,  2690,    30,   533,   634,  2199,    28,  5283,   654,   325,  5657,   288,  8039,
           480,  6020, 13587,  1568,   288,  3096,   282,   913,    28,  2030,  4345,   260,  8252,   282,  3961,  5574,
           284,  5136,    30,  1116,    35,    30, 18308, 11678, 31858,    42,  2287,   763,  4764,   284,  4340, 11228,
          3077,   416,   597,  1728,   354,  2081,   281, 11678, 31858,    28,   527,   416,  3055,  6020,   284,  2369,
         11783,    30,   669,   416,   919,   357,  8643,   327,  5283,   288,  2690,  2001,  3961,  2690,   288,  2220,
          3516,    28,   527,   416,  1022,   288, 20659,   284,  2061,  5770,    30,  1116,    36,    30, 16691,   281,
          2969,  1920,    42,  7840,  1363,   314,   597,  4439,  1971,   281,   260, 11706,   284,  3695,   282,  2969,
         12064,    30,   669,   416,  2151,   260,  8252,   282,  1647,  5574,   284,  5136,   281,  1642,    29, 11795,
          2429,    28,  2117,   585,   623,  2429,  6654,   335,   253,  6260,  3175,   282,  6020,   338,   359,  5857,
           288,  1971,   281,  2969,  1920,    30,   198,   198, 20832,    28,  2412,  1363,   314,  1953,   253,  1546,
          1645,   335,   260,  8252,   282,  3961,  5574,   284,  5136,   281,  1642,    29, 11795,  2429,    30,  1428,
          2498,  2353,   288,  2369,   653,  7499,  2811,  4562,   284,  1199,  2412,    29, 24914,  5995,  2718,    28,
           392,   416,   724,  2381,   338,  2573,   553,  1594,   288,  2458,   284, 15627,  3111,    28,  7142,   282,
           480,  4546,    30,     2,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
          -100,  -100,  -100,  -100,  -100,  -100,  -100, 42686,  8234,    17,  7840,  1363,   597,  5547,   260,  6367,
           284,  4759,   282,  3961,  2690,    28,   527,   416,   457,   253,  1546,  1645,   335,   624,  8252,   284,
         29497,   281,  1642,    29, 11795,  2429,    30,  3726,   359,   253,  1443,  1853,   281,   527,   451,   416,
          4317,    42,  1116,    33,    30, 18607, 23551,    42, 31928,  3947,  2466,   702, 15515,    28, 21834,    28,
           284, 24670,   416,  3055,  6367,  5730,   702,  8364,    28, 12223,    28,   284, 14796,    30,   669,   416,
           919,   357,  1990,   327,  3961,  2690,   288,   325, 15326,   429,  2969,  4286,   288,  1642,    29, 11795,
          2429,    30,   533,  1706,    28,  3977,  5770,   654,  2081,  1568,   288,  6367, 23551,    28,  2899,   288,
          2061,  5770,   327,  6461,    30,   216,    34,    30, 14576,   282,  4759,  5834,    42, 31928,  3947,  2466,
           416,   597,  3055, 45185,  4759,  5834,    28,   527,   359,  1895,   327,  8314,  3961,  2690,    30,  1550,
           623,  5834,   359,  3590,    28,   357,   416,   325,   540,  1990,   288,  3317,   284,  4249,  3961,  2690,
            28,  2899,   288, 48595,   284,  3368,    30,   669,   416,   919,  3961,  5574,   284,  5136,   540,  6280,
           284,  1181,  1770,   281,  1642,    29, 11795,  2429,    30,   216,    35,    30, 18308,  1114,  3368,    42,
          7840,  1363,   416,   597,  1022,   288,  2420,  1114,  3368,    28,   527,   553,   253,  1739,  1645,   335,
           260,  8252,   284, 29497,   282,  3961,  2690,    30, 31928,  3947,  2466,   416,  3055,  6020,    28,  4439,
           601,   288,   685,   288,  3368,    30,   533,  1706,    28,   645,  3961,  2690,   314,   441,  5000,  6314,
            28,   357,   416, 28255,   284,   685,   288,  3368,    30,   669,   416,  1022,   288, 20659,   284,  2061,
          5770,   327,  6461,    28,  2117,   281,  1642,    29, 11795,  2429,   837,  1594,   288,  3961,  2690,   314,
          2221,  3435,    30, 18813,    28,  2412,  1363,  5547,   260,  6367,   284,  4759,   282,  3961,  2690,   281,
          1545,  1853,    28,   527,   416,   457,   253,  1739,  1645,   335,   624,  8252,   284, 29497,   281,  1642,
            29, 11795,  2429,    30,  1428,  2498,  2353,   288,  2369,   653,  7499,  2811,  4562,   284,  1748,   281,
          2412,    29,   375, 13727,  5730,    28,   392,   416,   724,  2381,   338,  2573,   553,  1594,   288,  2458,
           284, 15627,  3111,    28,  7142,   282,   480,  4546,    30,     2,  -100,  -100,  -100,  -100,  -100,  -100,
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
          -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100]]), 
          'position_ids': torch.tensor([[   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,
           18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,
           36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,   51,   52,   53,
           54,   55,   56,   57,   58,   59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,
           72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,   85,   86,   87,   88,   89,
           90,   91,   92,   93,   94,   95,   96,   97,   98,   99,  100,  101,  102,  103,  104,  105,  106,  107,
          108,  109,  110,  111,  112,  113,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,  124,  125,
          126,  127,  128,  129,  130,  131,  132,  133,  134,  135,  136,  137,  138,  139,  140,  141,  142,  143,
          144,  145,  146,  147,  148,  149,  150,  151,  152,  153,  154,  155,  156,  157,  158,  159,  160,  161,
          162,  163,  164,  165,  166,  167,  168,  169,  170,  171,  172,  173,  174,  175,  176,  177,  178,  179,
          180,  181,  182,  183,  184,  185,  186,  187,  188,  189,  190,  191,  192,  193,  194,  195,  196,  197,
          198,  199,  200,  201,  202,  203,  204,  205,  206,  207,  208,  209,  210,  211,  212,  213,  214,  215,
          216,  217,  218,  219,  220,  221,  222,  223,  224,  225,  226,  227,  228,  229,  230,  231,  232,  233,
          234,  235,  236,  237,  238,  239,  240,  241,  242,  243,  244,  245,  246,  247,  248,  249,  250,  251,
          252,  253,  254,  255,  256,  257,  258,  259,  260,  261,  262,  263,  264,  265,  266,  267,  268,  269,
          270,  271,  272,  273,  274,  275,  276,  277,  278,  279,  280,  281,  282,  283,  284,  285,  286,  287,
          288,  289,  290,  291,  292,  293,  294,  295,  296,  297,  298,  299,  300,  301,  302,  303,  304,  305,
          306,  307,  308,  309,  310,  311,  312,  313,  314,  315,  316,  317,  318,  319,  320,  321,  322,  323,
          324,  325,  326,  327,  328,  329,  330,  331,  332,  333,  334,  335,  336,  337,  338,  339,  340,  341,
          342,  343,  344,  345,  346,  347,  348,  349,  350,  351,  352,  353,  354,  355,  356,  357,  358,  359,
          360,  361,  362,  363,  364,  365,  366,  367,  368,  369,  370,  371,  372,  373,  374,  375,  376,  377,
          378,  379,  380,  381,  382,  383,  384,  385,  386,  387,  388,  389,  390,  391,  392,  393,  394,  395,
          396,  397,  398,  399,  400,  401,  402,  403,  404,  405,  406,  407,  408,  409,  410,  411,  412,  413,
          414,  415,  416,  417,  418,  419,  420,  421,  422,  423,  424,  425,  426,  427,  428,  429,  430,  431,
          432,  433,  434,  435,  436,  437,  438,  439,  440,  441,  442,  443,  444,  445,  446,  447,  448,  449,
          450,  451,  452,  453,  454,  455,  456,  457,  458,  459,  460,  461,  462,  463,  464,  465,  466,  467,
          468,  469,  470,  471,  472,  473,  474,  475,  476,  477,  478,  479,  480,  481,  482,  483,  484,  485,
          486,  487,  488,  489,  490,  491,  492,  493,  494,  495,  496,  497,  498,  499,  500,  501,  502,  503,
          504,  505,  506,  507,  508,  509,  510,  511,  512,  513,  514,  515,  516,  517,  518,  519,  520,  521,
          522,  523,  524,  525,  526,  527,  528,  529,  530,  531,  532,  533,  534,  535,  536,  537,  538,  539,
          540,  541,  542,  543,  544,  545,  546,  547,  548,  549,  550,  551,  552,  553,  554,  555,  556,  557,
          558,  559,  560,  561,  562,  563,  564,  565,  566,  567,  568,  569,  570,  571,  572,  573,  574,  575,
          576,  577,  578,  579,  580,  581,  582,  583,  584,  585,  586,  587,  588,  589,  590,  591,  592,  593,
          594,  595,  596,  597,  598,  599,  600,  601,  602,  603,  604,  605,  606,  607,  608,  609,  610,  611,
          612,  613,  614,  615,  616,  617,  618,  619,  620,  621,  622,  623,  624,  625,  626,  627,  628,  629,
          630,  631,  632,  633,  634,  635,  636,  637,  638,  639,  640,  641,  642,  643,  644,  645,  646,  647,
          648,  649,  650,  651,  652,  653,  654,  655,  656,  657,  658,  659,  660,  661,  662,  663,  664,  665,
          666,  667,  668,  669,  670,  671,  672,  673,  674,  675,  676,  677,  678,  679,  680,  681,  682,  683,
          684,  685,  686,  687,  688,  689,  690,  691,  692,  693,  694,  695,  696,  697,  698,  699,  700,  701,
          702,  703,  704,  705,  706,  707,  708,  709,  710,  711,  712,  713,  714,  715,  716,  717,  718,  719,
          720,  721,  722,  723,  724,  725,  726,  727,  728,  729,  730,  731,  732,  733,  734,  735,  736,  737,
          738,  739,  740,  741,  742,  743,  744,  745,  746,  747,  748,  749,  750,  751,  752,  753,  754,  755,
          756,  757,  758,  759,  760,  761,  762,  763,  764,  765,  766,  767,  768,  769,  770,  771,  772,  773,
          774,  775,  776,  777,  778,  779,  780,  781,  782,  783,  784,  785,  786,  787,  788,  789,  790,  791,
          792,  793,  794,  795,  796,  797,  798,  799,  800,  801,  802,  803,  804,  805,  806,  807,  808,  809,
          810,  811,  812,  813,  814,  815,  816,  817,  818,  819,  820,  821,  822,  823,  824,  825,  826,  827,
          828,  829,  830,  831,  832,  833,  834,  835,  836,  837,  838,  839,  840,  841,  842,  843,  844,  845,
          846,  847,  848,  849,  850,  851,  852,  853,  854,  855,  856,  857,  858,  859,  860,  861,  862,  863,
          864,  865,  866,  867,  868,  869,  870,  871,  872,  873,  874,  875,  876,  877,  878,  879,  880,  881,
          882,  883,  884,  885,  886,  887,  888,  889,  890,  891,  892,  893,  894,  895,  896,  897,  898,  899,
          900,  901,  902,  903,  904,  905,  906,  907,  908,  909,  910,  911,  912,  913,  914,  915,  916,  917,
          918,  919,  920,  921,  922,  923,  924,  925,  926,  927,  928,  929,  930,  931,  932,  933,  934,  935,
          936,  937,  938,  939,  940,  941,  942,  943,  944,  945,  946,  947,  948,  949,  950,  951,  952,  953,
          954,  955,  956,  957,  958,  959,  960,  961,  962,  963,  964,  965,  966,  967,  968,  969,  970,  971,
          972,  973,  974,  975,  976,  977,  978,  979,  980,  981,  982,  983,  984,  985,  986,  987,  988,  989,
          990,  991,  992,  993,  994,  995,  996,  997,  998,  999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
         1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
         1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043,
         1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061,
         1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079,
         1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097,
         1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115,
         1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133,
         1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151,
         1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169,
         1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187,
         1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205,
         1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223,
         1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241,
         1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259,
         1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277,
         1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295,
         1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313,
         1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331,
         1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349,
         1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367,
         1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385,
         1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403,
         1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411,    0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
           10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,
           28,   29,   30,   31,   32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,
           46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,   63,
           64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,   79,   80,   81,
           82,   83,   84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   94,   95,   96,   97,   98,   99,
          100,  101,  102,  103,  104,  105,  106,  107,  108,  109,  110,  111,  112,  113,  114,  115,  116,  117,
          118,  119,  120,  121,  122,  123,  124,  125,  126,  127,  128,  129,  130,  131,  132,  133,  134,  135,
          136,  137,  138,  139,  140,  141,  142,  143,  144,  145,  146,  147,  148,  149,  150,  151,  152,  153,
          154,  155,  156,  157,  158,  159,  160,  161,  162,  163,  164,  165,  166,  167,  168,  169,  170,  171,
          172,  173,  174,  175,  176,  177,  178,  179,  180,  181,  182,  183,  184,  185,  186,  187,  188,  189,
          190,  191,  192,  193,  194,  195,  196,  197,  198,  199,  200,  201,  202,  203,  204,  205,  206,  207,
          208,  209,  210,  211,  212,  213,  214,  215,  216,  217,  218,  219,  220,  221,  222,  223,  224,  225,
          226,  227,  228,  229,  230,  231,  232,  233,  234,  235,  236,  237,  238,  239,  240,  241,  242,  243,
          244,  245,  246,  247,  248,  249,  250,  251,  252,  253,  254,  255,  256,  257,  258,  259,  260,  261,
          262,  263,  264,  265,  266,  267,  268,  269,  270,  271,  272,  273,  274,  275,  276,  277,  278,  279,
          280,  281,  282,  283,  284,  285,  286,  287,  288,  289,  290,  291,  292,  293,  294,  295,  296,  297,
          298,  299,  300,  301,  302,  303,  304,  305,  306,  307,  308,  309,  310,  311,  312,  313,  314,  315,
          316,  317,  318,  319,  320,  321,  322,  323,  324,  325,  326,  327,  328,  329,  330,  331,  332,  333,
          334,  335,  336,  337, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763,
         1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781,
         1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791],
        [   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,
           18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,
           36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,   51,   52,   53,
           54,   55,   56,   57,   58,   59,   60,   61,   62,   63,   64,   65,   66,   67,   68,   69,   70,   71,
           72,   73,   74,   75,   76,   77,   78,   79,   80,   81,   82,   83,   84,   85,   86,   87,   88,   89,
           90,   91,   92,   93,   94,   95,   96,   97,   98,   99,  100,  101,  102,  103,  104,  105,  106,  107,
          108,  109,  110,  111,  112,  113,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,  124,  125,
          126,  127,  128,  129,  130,  131,  132,  133,  134,  135,  136,  137,  138,  139,  140,  141,  142,  143,
          144,  145,  146,  147,  148,  149,  150,  151,  152,  153,  154,  155,  156,  157,  158,  159,  160,  161,
          162,  163,  164,  165,  166,  167,  168,  169,  170,  171,  172,  173,  174,  175,  176,  177,  178,  179,
          180,  181,  182,  183,  184,  185,  186,  187,  188,  189,  190,  191,  192,  193,  194,  195,  196,  197,
          198,  199,  200,  201,  202,  203,  204,  205,  206,  207,  208,  209,  210,  211,  212,  213,  214,  215,
          216,  217,  218,  219,  220,  221,  222,  223,  224,  225,  226,  227,  228,  229,  230,  231,  232,  233,
          234,  235,  236,  237,  238,  239,  240,  241,  242,  243,  244,  245,  246,  247,  248,  249,  250,  251,
          252,  253,  254,  255,  256,  257,  258,  259,  260,  261,  262,  263,  264,  265,  266,  267,  268,  269,
          270,  271,  272,  273,  274,  275,  276,  277,  278,  279,  280,  281,  282,  283,  284,  285,  286,  287,
          288,  289,  290,  291,  292,  293,  294,  295,  296,  297,  298,  299,  300,  301,  302,  303,  304,  305,
          306,  307,  308,  309,  310,  311,  312,  313,  314,  315,  316,  317,  318,  319,  320,  321,  322,  323,
          324,  325,  326,  327,  328,  329,  330,  331,  332,  333,  334,  335,  336,  337,  338,  339,  340,  341,
          342,  343,  344,  345,  346,  347,  348,  349,  350,  351,  352,  353,  354,  355,  356,  357,  358,  359,
          360,  361,  362,  363,  364,  365,  366,  367,  368,  369,  370,  371,  372,  373,  374,  375,  376,  377,
          378,  379,  380,  381,  382,  383,  384,  385,  386,  387,  388,  389,  390,  391,  392,  393,  394,  395,
          396,  397,  398,  399,  400,  401,  402,  403,  404,  405,  406,  407,  408,  409,  410,  411,  412,  413,
          414,  415,  416,  417,  418,  419,  420,  421,  422,  423,  424,  425,  426,  427,  428,  429,  430,  431,
          432,  433,  434,  435,  436,  437,  438,  439,  440,  441,  442,  443,  444,  445,  446,  447,  448,  449,
          450,  451,  452,  453,  454,  455,  456,  457,  458,  459,  460,  461,  462,  463,  464,  465,  466,  467,
          468,  469,  470,  471,  472,  473,  474,  475,  476,  477,  478,  479,  480,  481,  482,  483,  484,  485,
          486,  487,  488,  489,  490,  491,  492,  493,  494,  495,  496,  497,  498,  499,  500,  501,  502,  503,
          504,  505,  506,  507,  508,  509,  510,  511,  512,  513,  514,  515,  516,  517,  518,  519,  520,  521,
          522,  523,  524,  525,  526,  527,  528,  529,  530,  531,  532,  533,  534,  535,  536,  537,  538,  539,
          540,  541,  542,  543,  544,  545,  546,  547,  548,  549,  550,  551,  552,  553,  554,  555,  556,  557,
          558,  559,  560,  561,  562,  563,  564,  565,  566,  567,  568,  569,  570,  571,  572,  573,  574,  575,
          576,  577,  578,  579,  580,  581,  582,  583,  584,  585,  586,  587,  588,  589,  590,  591,  592,  593,
          594,  595,  596,  597,  598,  599,  600,  601,  602,  603,  604,  605,  606,  607,  608,  609,  610,  611,
          612,  613,  614,  615,  616,  617,  618,  619,  620,  621,  622,  623,  624,  625,  626,  627,  628,  629,
          630,  631,  632,  633,  634,  635,  636,  637,  638,  639,  640,  641,  642,  643,  644,  645,  646,  647,
          648,  649,  650,  651,  652,  653,  654,  655,  656,  657,  658,  659,  660,  661,  662,  663,  664,  665,
          666,  667,  668,  669,  670,  671,  672,  673,  674,  675,  676,  677,  678,  679,  680,  681,  682,  683,
          684,  685,  686,  687,  688,  689,  690,  691,  692,  693,  694,  695,  696,  697,  698,  699,  700,  701,
          702,  703,  704,  705,  706,  707,  708,  709,  710,  711,  712,  713,  714,  715,  716,  717,  718,  719,
          720,  721,  722,  723,  724,  725,  726,  727,  728,  729,  730,  731,  732,  733,  734,  735,  736,  737,
          738,  739,  740,  741,  742,  743,  744,  745,  746,  747,  748,  749,  750,  751,  752,  753,  754,  755,
          756,  757,  758,  759,  760,  761,  762,  763,  764,  765,  766,  767,  768,  769,  770,  771,  772,  773,
          774,  775,  776,  777,  778,  779,  780,  781,  782,  783,  784,  785,  786,  787,  788,  789,  790,  791,
          792,  793,  794,  795,  796,  797,  798,  799,  800,  801,  802,  803,  804,  805,  806,  807,  808,  809,
          810,  811,  812,  813,  814,  815,  816,  817,  818,  819,  820,  821,  822,  823,  824,  825,  826,  827,
          828,  829,  830,  831,  832,  833,  834,  835,  836,  837,  838,  839,  840,  841,  842,  843,  844,  845,
          846,  847,  848,  849,  850,  851,  852,  853,  854,  855,  856,  857,  858,  859,  860,  861,  862,  863,
          864,  865,  866,  867,  868,  869,  870,  871,  872,  873,  874,  875,  876,  877,  878,  879,  880,  881,
          882,  883,  884,  885,  886,  887,  888,  889,  890,  891,  892,  893,  894,  895,  896,  897,  898,  899,
          900,  901,  902,  903,  904,  905,  906,  907,  908,  909,  910,  911,  912,  913,  914,  915,  916,  917,
          918,  919,  920,  921,  922,  923,  924,  925,  926,  927,  928,  929,  930,  931,  932,  933,  934,  935,
          936,  937,  938,  939,  940,  941,  942,  943,  944,  945,  946,  947,  948,  949,  950,  951,  952,  953,
          954,  955,  956,  957,  958,  959,  960,  961,  962,  963,  964,  965,  966,  967,  968,  969,  970,  971,
          972,  973,  974,  975,  976,  977,  978,  979,  980,  981,  982,  983,  984,  985,  986,  987,  988,  989,
          990,  991,  992,  993,  994,  995,  996,  997,  998,  999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007,
         1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025,
         1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043,
         1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061,
         1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079,
         1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097,
         1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115,
         1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133,
         1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151,
         1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169,
         1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187,
         1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205,
         1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223,
         1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241,
         1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259,
         1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277,
         1278, 1279, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295,
         1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313,
         1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331,
         1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349,
         1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367,
         1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385,
         1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403,
         1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421,
         1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439,
         1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457,
         1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475,
         1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493,
         1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511,
         1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529,
         1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547,
         1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565,
         1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583,
         1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601,
         1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619,
         1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637,
         1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655,
         1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673,
         1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691,
         1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709,
         1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1726, 1727,
         1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745,
         1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763,
         1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781,
         1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791]])}


        # XXX: probably need to do padding so that all sequence chunks are the same?!
        import math
        print(f"{len(batch['input_ids'][0])=}")
        print(f"{len(batch['input_ids'][1])=}")
        seq_length = len(batch["input_ids"][0])

        sp_world_size = groups._get_sequence_parallel_world_size()
        sp_rank = groups._get_sequence_parallel_rank()
        chunk_len = math.ceil(seq_length / sp_world_size)
        print(f"{seq_length=}")
        print(f"{chunk_len=}")
        for k in batch.keys():
            batch[k] = batch[k][:, chunk_len*sp_rank:chunk_len*(sp_rank+1)].to(self.device)
            print(f"{batch[k].shape=}")
        #import sys; sys.exit(0)

        outputs = self.model.generate(**batch, do_sample=False)
        #print(outputs)
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #if self.global_rank == 0:
        
        chunk = decoded[0][-100:].replace('\n',' ')
        print(f"RANK {self.global_rank}: {chunk}")
        import sys
        sys.exit(0)

        self.model.train()
        loss = self.loss(batch)
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
