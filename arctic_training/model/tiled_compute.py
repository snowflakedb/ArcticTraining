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


"""

The core autograd function sequence_tiled_compute lives in Deepspeed, here we have applied versions that use it.

"""

import math

import torch
import torch.distributed as dist
from transformers import AutoConfig

from deepspeed.runtime.sequence_parallel.ulysses_sp import SequenceTiledCompute
from deepspeed.runtime.sequence_parallel.ulysses_sp import sequence_tiled_compute


def get_model_type(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)
    return config.model_type


import torch.nn as nn


# simplified MLP with just one dummy weight
class MyLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dummy_proj2 = nn.Linear(self.hidden_size, self.hidden_size * 2, bias=False)
        self.dummy_proj = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, x):
        return self.dummy_proj(self.dummy_proj2(x))


# import torch.nn as nn
# from transformers.activations import ACT2FN
# class TiledLlamaMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
#         self.act_fn = ACT2FN[config.hidden_act]
#         #self.dummy_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.mlp_bias)

#     def real_forward(self, x):
#         down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
#         return down_proj

#     def forward(self, x):
#         from functools import partial
#         #print("Hello")
#         return SequenceTiledCompute.apply(
#             self.real_forward,
#             x,
#         )


def tiled_mlp_forward_common(self, x):
    """a monkey patch to replace modeling_llama.LlamaMLP.forward and other identical MLP implementations to perform a tiled compute of the same"""
    # import os
    # rank = int(os.getenv("LOCAL_RANK", 0))
    # if rank == 0:
    #     print(f"computing main {x.shape}")

    # XXX: temp
    # num_shards = 6
    num_shards = "auto"

    if num_shards == "auto":
        bs, seqlen, hidden = x.shape
        # XXX: not too many?
        num_shards = math.ceil(seqlen / hidden)

        # it's crucial that all ranks run the same number of shards, otherwise if one of the ranks runs less shards, there will be a deadlock as that rank will stop running sooner than others and will not supply its weights shard to other ranks. So we will use the max value across all ranks.
        # XXX: but this will run on every layer - it'd be good to cache the number of shards as it doesn't change during the iteration
        tensor = torch.tensor(num_shards, device=x.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        num_shards = tensor.item()
        # print(f"derived {num_shards} for {seqlen=} and {hidden=} max'ed across ranks")

    # print(f"{self.down_proj.weight.shape=}")
    # print(f"{self.up_proj.weight.shape=}")
    kwargs_to_shard = dict(x=x)
    kwargs_to_pass = dict(self=self)
    # kwargs_to_pass = dict(
    #     down=self.down_proj.weight,
    #     gate=self.gate_proj.weight,
    #     up=self.up_proj.weight,
    #     # down=self.down_proj.__call__,
    #     # gate=self.gate_proj.__call__,
    #     # up=self.up_proj.__call__,
    #     act_fn=self.act_fn,
    # )
    # kwargs_to_pass = dict(self=self.dummy_proj.weight)
    grad_requiring_tensor_key = "x"
    compute_params = [self.down_proj.weight, self.gate_proj.weight, self.up_proj.weight]
    seqlen = x.shape[1]

    def mlp_forward2(x, down, gate, up, act_fn):
        # return down(act_fn(gate(x)) * up(x))
        # print(f"{x.shape=}")
        # print(f"{up.shape=}")

        a = x @ up.t()
        b = act_fn(x @ gate.t())
        c = a * b
        return c @ down.t()

    def mlp_forward(x, self):  # self=None, x=None):
        # print(f"mlp_forward computing sub {x.shape}")
        # return self.dummy_proj(x)

        # leaks the size of mm
        # return x @ self
        # no leak
        # return x*2
        # return self.down_proj(self.gate_proj(x))
        # y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    return Compute2.apply(
        mlp_forward,
        x,
        self,
        seqlen,
        num_shards,
        compute_params,
    )

    # from functools import partial
    # return SequenceTiledCompute.apply(
    #     mlp_forward2,
    #     x,
    #     self.down_proj.weight,
    #     self.gate_proj.weight,
    #     self.up_proj.weight,
    #     self.act_fn,
    # )

    # def mlp_forward(self=None, x=None):
    #     #print(f"mlp_forward computing sub {x.shape}")

    #     # leaks the size of mm
    #     return x @ self
    #     # no leak
    #     return x*2
    #     #return self.down_proj(self.gate_proj(x))
    #     #y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    #     #return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    # return SequenceTiledCompute.apply(
    #     mlp_forward,
    #     x,
    #     self.dummy_proj.weight,
    # )

    x = sequence_tiled_compute(
        mlp_forward,
        seqlen,
        num_shards,
        kwargs_to_shard,
        kwargs_to_pass,
        grad_requiring_tensor_key,
        compute_params,
        output_unshard_dimension=1,  # x
        output_reduction=None,
    )
    return x


def mlp_forward_new(self, x):

    def mlp_forward(x, self):
        # return self.dummy_proj(self.dummy_proj2(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # return self.dummy_proj(x)

    print(f"{x.shape=}")
    # print(f"{self.dummy_proj.weight.shape=}")

    # mlp_forward(x, self)

    return Compute.apply(mlp_forward, x, self)


class Compute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fn,
        x,
        self,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.self = self
        ctx.save_for_backward(x)

        with torch.no_grad():
            z = fn(x, self)
            return fn(x, self)

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        self = ctx.self

        x1 = x.detach()
        x1.requires_grad = x.requires_grad
        with torch.enable_grad():
            output = fn(x1, self)
            output = fn(x1, self)

        # hooks will start firing here - prints will show up here
        torch.autograd.backward(output, grads[0])
        return (None, x1.grad, None)


class Compute2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fn,
        x,
        self,
        seqlen,
        shards,
        compute_params,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.self = self
        ctx.seqlen = seqlen
        ctx.shards = shards
        ctx.compute_params = compute_params
        ctx.save_for_backward(x)

        with torch.no_grad():
            shard_step = math.ceil(seqlen / shards)
            output_shards = []

            for i in range(shards):
                output = fn(x[:, i * shard_step : (i + 1) * shard_step], self)
                output_shards.append(output)
            output_unsharded = torch.cat(output_shards, dim=1)

            return output_unsharded

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        self = ctx.self
        seqlen = ctx.seqlen
        shards = ctx.shards
        compute_params = ctx.compute_params

        x1 = x.detach()
        x1.requires_grad = x.requires_grad
        x = x1

        incoming_grad = grads[0]
        x_grad = torch.zeros_like(x)
        x_shards = list(torch.chunk(x, chunks=shards, dim=1))

        shard_step = x_shards[0].numel()
        for i in range(shards):

            if compute_params is not None:
                if i + 1 < shards:
                    for param in compute_params:
                        param.ds_grad_is_ready = False
                else:
                    # last shard, can add the grad
                    for param in compute_params:
                        param.ds_grad_is_ready = True

            x_shard = x_shards.pop(0)

            shard_offset = i * shard_step
            x_shard.grad = x_grad.view(-1).narrow(0, shard_offset, x_shard.numel()).view_as(x_shard)

            # make it optional
            with torch.enable_grad():
                output = fn(x_shard, self)

                incoming_grad_shard = incoming_grad.view(-1).narrow(0, shard_offset, x_shard.numel()).view_as(x_shard)
                torch.autograd.backward(output, incoming_grad_shard)

        x_grad /= shards

        return (None, x.grad, None, None, None, None)


def enable_tiled_mlp_compute(model_name_or_path):
    """
    Important: this monkey patching call, that overrides the original HF Transformers model's MLP class, has to happen before model is instantiated.
    Currently only some models are supported, but we can easily add support for more model architectures if needed.

    Also beware of other packages overriding it - e.g. Liger-Kernel - you can tell Liger-Kernel not to override it via `from_pretrained(..., swiglu=False)`
    """

    model_type = get_model_type(model_name_or_path)
    if model_type == "llama":
        from transformers.models.llama import modeling_llama

        # modeling_llama.LlamaMLP = MyLlamaMLP
        # modeling_llama.LlamaMLP.forward = mlp_forward_new
        modeling_llama.LlamaMLP.forward = tiled_mlp_forward_common
    elif model_type == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2

        modeling_qwen2.Qwen2MLP.forward = tiled_mlp_forward_common
    elif model_type == "qwen3":
        from transformers.models.qwen3 import modeling_qwen3

        modeling_qwen3.Qwen3MLP.forward = tiled_mlp_forward_common
    else:
        raise ValueError(
            f"model type {model_type} is currently not supported. Please open an issue and ask to add Tiled MLP"
            f" support for {model_type}."
        )

