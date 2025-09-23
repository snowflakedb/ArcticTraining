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

import re
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Union
from typing import cast

import torch
from torch import nn

from arctic_training.model.moe.moe import ArcticMoE


def detect_if_moe_model(model):
    return any(k for k in model.config.__dict__.keys() if re.search("experts", k))


def remap_moe_mlp_params_to_deepspeed_moe(model):
    """
    expects an unwrapped model object
    """

    device = model.device

    config = model.config
    import copy

    c = copy.copy(config)

    # remap config entries
    if hasattr(config, "num_local_experts"):  # gpt-oss
        c.num_experts = config.num_local_experts
    elif hasattr(config, "num_experts"):  # qwen
        c.num_experts = config.num_experts
    else:
        raise ValueError(
            "Can't find an entry for number of experts in model's config. The config object has the following keys:"
            f" {config.__dict__.keys()}"
        )

    c.model_dim = config.hidden_size
    c.intermediate_dim = config.intermediate_size
    c.input_dtype = model.dtype
    c.activation = config.hidden_act
    c.top_k = config.num_experts_per_tok

    # XXX: need a new yaml config?
    c.use_triton = False

    for layer_num, layer_module in enumerate(model.model.layers):
        # some models don't have moe in every layer
        if not hasattr(layer_module.mlp, "experts"):
            print(f"{layer_num} is not an MoE layer")
            continue

        # XXX: is there a point of using meta-device?
        with torch.device("meta"):
            arctic_moe = ArcticMoE(c)
        # move onto cuda
        arctic_moe.to_empty(device=device)

        # [n for n, _ in arctic_moe.named_parameters()]
        # ['expert_intermediate_weights', 'expert_output_weights', '_gate_proj.weight']
        #
        # [n for n, _ in m.model.layers[0].mlp.experts.named_parameters()]
        # gpt-oss
        # ['gate_up_proj', 'gate_up_proj_bias', 'down_proj', 'down_proj_bias']
        # qwen
        # XXX
        orig_experts = layer_module.mlp.experts
        # qwne is a list
        experts_is_a_list = True if isinstance(orig_experts, torch.nn.modules.container.ModuleList) else False

        def copy_weights(from_name, to_param):
            if experts_is_a_list:  # ModuleList models like qwen
                weight_stacked = torch.stack(
                    [getattr(orig_experts[i], from_name).weight for i in range(len(orig_experts))]
                )
            else:  # gpt-oss-like models with a stack of experts weights
                weight_stacked = getattr(orig_experts, from_name).weight
            to_param.copy_(weight_stacked)

        with torch.no_grad():
            for n, m in orig_experts.named_parameters():
                if n == "gate_up_proj":  # gpt-oss
                    gate_up = m
                    # XXX: this is wrong/incomplete
                    # gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                    # arctic_moe._gate_proj.weight.copy_(gate)
                    # arctic_moe.expert_intermediate_weights.copy_(up)

                    # move for now
                    arctic_moe.gate_up = gate_up
                elif n == "gate_proj":
                    copy_weights("gate_proj", arctic_moe._gate_proj.weight)
                elif n == "up_proj":
                    copy_weights("up_proj", arctic_moe.expert_intermediate_weights)
                elif n == "down_proj":
                    copy_weights("down_proj", arctic_moe.expert_output_weights)

        # override
        layer_module.mlp = arctic_moe

        print(f"{layer_module.mlp}")

    print(f"Rewritten model: {model}")

    # now copy over the params from the original model, while freeing their memory usage


"""
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("tiny-random/gpt-oss-bf16")
model.model.layers[0].mlp.experts.gate_up_proj.shape
Out[5]: torch.Size([32, 32, 128])

from arctic_training.model.moe.moe import ArcticMoE
arctic_moe = ArcticMoE(c)
arctic_moe._gate_proj.weight.shape

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("snake7gun/tiny-random-qwen3moe")
model.model.layers[0].mlp.experts.gate_proj.shape
Out[5]: torch.Size([32, 32, 128])





"""


# def find_mlp_module(module):
#     # print(f"{module=}")
#     for n, module in module.named_children():
#         # print(f"{n=}")
#         if n == "mlp":
#             return module
#         else:
#             mlp_module = find_mlp_module(module)
#             if mlp_module is not None:
#                 return mlp_module

#     return None

# mlp = find_mlp_module(model)  # noqa

# model.layers.0.mlp
# model.layers.0.mlp.router
# model.layers.0.mlp.experts
# model.layers.1.mlp
# model.layers.1.mlp.router
# model.layers.1.mlp.experts


def identify_moe_params(model, expert_parallel_size):
    # regex = r'deepspeed_moe.experts.deepspeed_experts.*?.weight'
    # regex = r"experts"
    # moe_param_data_ptrs = [p.data_ptr for n, p in model.named_parameters() if re.search(regex, n) is not None]

    expert_group_name = f"ep_size_{expert_parallel_size}"

    moe_param_data_ptrs = []
    for n, m in model.named_modules():
        if isinstance(m, ArcticMoE):
            moe_param_data_ptrs += [p.data_ptr for n, p in m.named_parameters()]
            for p in m.parameters():
                p.group_name = expert_group_name

    print(f"Found {len(moe_param_data_ptrs)} MoE params")
    # die
    return moe_param_data_ptrs


def is_moe_param(param: torch.Tensor, moe_param_data_ptrs: List) -> bool:
    # XXX: this will be set once MoE class is used
    if param.data_ptr in moe_param_data_ptrs and hasattr(param, "group_name"):
        # print(F"MoE param")
        return True
    else:
        return False


def split_params_into_different_moe_groups_for_optimizer(
    param_groups: Union[Dict[str, Any], Tuple[Dict[str, Any], ...], List[Dict[str, Any]]],
    moe_param_data_ptrs: List,
    max_group_size: Union[int, float] = 178956971,
) -> List[Dict[str, Any]]:
    """Split parameters into different MoE groups for optimizer

    Args:
        param_groups (Union[Dict[str, Any], Tuple[Dict[str, Any], ...], List[Dict[str, Any]]])
            The list of parameter groups to split

    Returns:
        List[Dict[str, Any]]:
        list of MoE/non-MoE groups for optimizer
    """
    if isinstance(param_groups, tuple):
        param_groups = list(param_groups)  # Tuple cannot be modified
    elif isinstance(param_groups, dict):
        param_groups = [param_groups]
    elif not isinstance(param_groups, list):
        raise ValueError(f"Unknown param group type of {type(param_groups)}")

    # print("Splitting into Moe")

    # gather all data parallel group names
    data_parallel_group_names: Set[str] = set()
    for param_group in param_groups:
        # for param in cast(List[nn.Parameter], param_group["params"]):
        for param in param_group["params"]:
            if is_moe_param(param, moe_param_data_ptrs):
                data_parallel_group_names.add(param.group_name)

    # Create the param MoE groups, leave param assign to next step
    group_moe: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for param_group in param_groups:
        for key in data_parallel_group_names:
            group_moe[param_group["name"]][key] = {
                **param_group,
                "name": key,
                "moe": True,
                "params": [],
            }

    # Assign param
    for param_group in param_groups:
        new_params: List[nn.Parameter] = []

        for param in cast(List[nn.Parameter], param_group["params"]):
            if is_moe_param(param, moe_param_data_ptrs):
                group_moe[param_group["name"]][param.group_name]["params"].append(param)
            else:
                new_params.append(param)
        param_group["params"] = new_params

    # Flatten the moe groups
    if max_group_size is not None:
        for moe_group in group_moe.values():
            for param_group in moe_group.values():
                cur_group: List[nn.Parameter] = []
                all_groups: List[List[nn.Parameter]] = []
                size_of_cur_group = 0

                for param in cast(List[nn.Parameter], param_group["params"]):
                    if size_of_cur_group + param.numel() <= max_group_size:
                        cur_group.append(param)
                        size_of_cur_group += param.numel()
                    else:
                        all_groups.append(cur_group)
                        cur_group = [param]
                        size_of_cur_group = param.numel()

                if cur_group:
                    all_groups.append(cur_group)

                for group in all_groups:
                    param_groups.append({**param_group, "params": group})
    else:
        for moe_group in group_moe.values():
            for param_group in moe_group.values():
                param_groups.append(param_group)

    return param_groups
