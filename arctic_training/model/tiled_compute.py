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


import math

import torch
import torch.distributed as dist
from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP
from transformers import AutoConfig


def get_model_type(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)
    return config.model_type


def tiled_mlp_forward_common(self, x):
    """a monkey patch to replace modeling_llama.LlamaMLP.forward and other identical MLP implementations to perform a tiled compute of the same"""

    num_shards = "auto"

    if num_shards == "auto":
        bs, seqlen, hidden = x.shape
        num_shards = math.ceil(seqlen / hidden)

        # it's crucial that all ranks run the same number of shards, otherwise if one of the ranks
        # runs fewer shards than the rest, there will be a deadlock as that rank will stop running
        # sooner than others and will not supply its ZeRO-3 weights shard to other ranks. So we
        # will use the max value across all ranks.
        #
        # XXX: but this will run on every layer - it'd be good to cache the number of shards as it
        # doesn't change during the iteration, but may change between iterations if seqlen is varlen
        tensor = torch.tensor(num_shards, device=x.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        num_shards = tensor.item()
        # print(f"derived {num_shards} for {seqlen=} and {hidden=} max'ed across ranks")

    compute_params = [self.down_proj.weight, self.gate_proj.weight, self.up_proj.weight]

    def mlp_forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    return TiledMLP.apply(
        mlp_forward,
        self,
        x,
        num_shards,
        compute_params,
    )


def enable_tiled_mlp_compute(model_name_or_path):
    """
    Important: this monkey patching call, that overrides the original HF Transformers model's MLP class, has to happen before a model is instantiated.

    Currently only some models are supported, but we can easily add support for more model architectures if needed.

    Also beware of other packages overriding it - e.g. Liger-Kernel - you can tell Liger-Kernel not to override it via its `from_pretrained(..., swiglu=False)`
    """

    model_type = get_model_type(model_name_or_path)
    if model_type == "llama":
        from transformers.models.llama import modeling_llama

        modeling_llama.LlamaMLP.forward = tiled_mlp_forward_common
    elif model_type == "qwen2":
        from transformers.models.qwen2 import modeling_qwen2

        modeling_qwen2.Qwen2MLP.forward = tiled_mlp_forward_common
    elif model_type == "qwen3":
        from transformers.models.qwen3 import modeling_qwen3

        modeling_qwen3.Qwen3MLP.forward = tiled_mlp_forward_common
    else:
        raise ValueError(
            f"model type {model_type} is currently not supported. Please open an Issue and ask to add Tiled MLP"
            f" support for {model_type} or alternatively submit a PR."
        )
