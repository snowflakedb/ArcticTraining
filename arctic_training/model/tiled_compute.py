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
from typing import Tuple

import torch
import torch.distributed as dist
from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def get_model_type(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)
    return config.model_type


def get_causal_lm_model_cls_prefix(model_type: str) -> Tuple[str, str]:
    if model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        causal_lm_cls = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]
        causal_lm_cls_prefix = causal_lm_cls
        for suffix in [
            "ForCausalLM",
            "ForConditionalGeneration",
            "LMHeadModel",
            "GenerationDecoder",
        ]:
            causal_lm_cls_prefix = causal_lm_cls_prefix.replace(suffix, "")
        return causal_lm_cls_prefix, causal_lm_cls
    causal_lm_cls_prefix = "".join([part.capitalize() for part in model_type.split("_")])
    return causal_lm_cls_prefix, f"{causal_lm_cls_prefix}ForCausalLM"


def tiled_mlp_forward_common(self, x):
    """a monkey patch to replace modeling_llama.LlamaMLP.forward and other identical MLP implementations to perform a tiled compute of the same"""

    num_shards = "auto"

    if num_shards == "auto":
        # x.shape could be [bs, seqlen, hidden_size] or [seqlen, hidden_size] (moe experts)
        seqlen, hidden = x.shape[-2:]
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

    This code will try to override the MLP's `forward` of any model. This of course will only work correctly as long as the MLP function is `down(act_fn(gate_proj(x)) * up_proj(x))` - the behavior is unpredictable if it's not.

    Also beware of other packages overriding it - e.g. Liger-Kernel - you can tell Liger-Kernel not to override it via its `from_pretrained(..., swiglu=False)`
    """

    model_type = get_model_type(model_name_or_path)
    try:
        # Dynamically import the module and MLP class
        module_path = f"transformers.models.{model_type}.modeling_{model_type}"
        model_cls_prefix, _ = get_causal_lm_model_cls_prefix(model_type)
        module = __import__(module_path, fromlist=[f"{model_cls_prefix}MLP"])
        mlp_cls = getattr(module, f"{model_cls_prefix}MLP")
        setattr(mlp_cls, "forward", tiled_mlp_forward_common)
    except Exception as e:
        raise ValueError(f"Failed to autodetect {mlp_cls}: {e}")
