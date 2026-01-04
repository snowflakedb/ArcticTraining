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

import copy
import os
import re
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple
from typing import Union
from typing import cast

import deepspeed.runtime.engine
import torch
import torch.distributed as dist
import torch.nn.functional as F
from deepspeed.checkpoint.utils import clone_tensors_for_torch_save
from deepspeed.runtime.checkpoint_engine import TorchCheckpointEngine
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.utils import groups
from torch import nn

from arctic_training.debug.utils import pr0
from arctic_training.debug.utils import see_memory_usage

# from arctic_training.debug.utils import tensor_has_nan
from arctic_training.model.moe.moe import ArcticMoE


def monkey_patch_ds_moe():

    # override the original Meg-DS profiler print util
    # from deepspeed.runtime.engine import DeepSpeedEngine

    from arctic_training.model.moe.moe import print_forward_breakdown

    DeepSpeedEngine.print_forward_breakdown = print_forward_breakdown

    # DS checkpointing
    DeepSpeedEngine._save_moe_checkpoint = amoe_save_checkpoint
    DeepSpeedEngine.load_moe_state_dict = amoe_load_state_dict


def detect_if_moe_model(model):
    return any(k for k in model.config.__dict__.keys() if re.search("experts", k))


@torch.no_grad()
def remap_orig_moe_mlp_params_to_arctic_moe(model, ep_size, is_resume):
    """
    remaps the existing model's mlp moe params to arctic_moe unified representation, modifying the model.

    Currently supporting:
    - full: Qwen3MoeForCausalLM, Qwen3NextForCausalLM
    - partial: GptOssForCausalLM (stopped syncing it for now)

    XXX: this will not work with zero.Init since the weights will be already sharded. If we want to add zero.Init support we will need to gather, remap, re-shard. But we have no Z3 support in DS-MoE anyway, so it's irrelevant until we do.

    Args:
      - model: expects an unwrapped model object
      - ep_size: EP size
      - is_resume: don't copy the weights on resume, they will be filled from a DS checkpoint at a later time
    """

    # this plugs us into the old DeepspeedMoE system whose integration into Z2 is needed for ArcticMoE to work with ZeRO-2
    # https://github.com/deepspeedai/DeepSpeed/blob/69e03e52d0ebc567d34a163e925899751f7dbcb8/deepspeed/runtime/engine.py#L1323
    deepspeed.runtime.engine.MoE = ArcticMoE

    pr0(f"Resuming mode: {is_resume}", force=True)

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    # model.to(device)
    # device = model.device
    meta_device = torch.device("meta")
    config = model.config

    arctic_moe_config = SimpleNamespace(
        **dict(
            model_dim=config.hidden_size,
            input_dtype=model.dtype,
            # activation=config.hidden_act,
            top_k=config.num_experts_per_tok,
            normalize_topk_scores=getattr(config, "norm_topk_prob", False),
            loss_coeff=config.router_aux_loss_coef,
        )
    )

    supported_activations = ["relu", "gelu", "silu"]
    if "GptOssForCausalLM" in config.architectures[0]:
        # gpt-oss uses a variation of silu with an alpha coefficient for the sigmoid arg
        # alpha comes from https://github.com/huggingface/transformers/blob/94df0e65602922be2831b3faa457a2bde78b936b/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L78C7-L78C27
        alpha = 1.702
        arctic_moe_config.act_fn = lambda x: x * torch.sigmoid(x * alpha)
    elif config.hidden_act in supported_activations:
        arctic_moe_config.act_fn = getattr(F, config.hidden_act)
    else:
        raise ValueError(f"Unsupported activation {config.hidden_act}")

    # some models have a different intermediate size for experts than normal mlp
    if hasattr(config, "moe_intermediate_size"):  # qwen-next
        arctic_moe_config.intermediate_dim = config.moe_intermediate_size
    else:
        arctic_moe_config.intermediate_dim = config.intermediate_size

    # pr0(f"{arctic_moe_config}", force=True)
    # remap config entries which use different names for the same concept
    if hasattr(config, "num_local_experts"):  # gpt-oss
        arctic_moe_config.num_experts = config.num_local_experts
    elif hasattr(config, "num_experts"):  # qwen
        arctic_moe_config.num_experts = config.num_experts
    else:
        raise ValueError(
            "Can't find an entry for number of experts in model's config. The config object has the following keys:"
            f" {config.__dict__.keys()}"
        )

    # XXX: this is the only signal I found so far in the qwen3-next model config - could probably check the model for presense of mlp.shared_expert
    if hasattr(config, "shared_expert_intermediate_size"):
        arctic_moe_config.use_shared_expert = True
    else:
        arctic_moe_config.use_shared_expert = False

    archs_with_router_scores = ["GptOssForCausalLM"]
    arctic_moe_config.return_router_scores = True if config.architectures[0] in archs_with_router_scores else False

    # XXX: need a new yaml config to use triton?
    arctic_moe_config.use_triton = False
    # at the moment the models we support are all gated
    arctic_moe_config.is_gated = True

    ep_group_name = f"ep_size_{ep_size}"
    ep_rank = groups._get_expert_parallel_rank(ep_group_name)
    ep_group = groups._get_expert_parallel_group(ep_group_name)
    num_local_experts = arctic_moe_config.num_experts // ep_size
    local_expert_indices = list(range(num_local_experts * ep_rank, num_local_experts * (ep_rank + 1)))

    arctic_moe_config.ep_size = ep_size
    arctic_moe_config.ep_rank = ep_rank
    arctic_moe_config.ep_group = ep_group

    pr0(f"Original model: {model}", force=True)

    layer = 0
    for layer_num, layer_module in enumerate(model.model.layers):
        # some models don't have moe in every layer
        if not hasattr(layer_module.mlp, "experts"):
            # pr0(f"{layer_num} is not an MoE layer", force=True)
            continue

        start = time.time()
        layer += 1
        see_memory_usage(f"{layer} start new moe layer", force=False)

        # pr0(f"{layer_num} is an MoE layer, force=True")
        # XXX: is there a point of using meta-device - it won't preallocate structures
        with meta_device:
            arctic_moe = ArcticMoE(arctic_moe_config)
        # move onto cuda
        arctic_moe.to_empty(device=device)
        see_memory_usage(f"{layer} after amoe layer created", force=False)

        # [n for n, _ in arctic_moe.named_parameters()]
        # ['expert_intermediate_weights', 'expert_output_weights', '_gate_proj.weight']
        #
        # [n for n, _ in m.model.layers[0].mlp.experts.named_parameters()]
        # gpt-oss
        # ['gate_up_proj', 'gate_up_proj_bias', 'down_proj', 'down_proj_bias']
        # qwen
        # XXX

        def copy_weights():

            # qwen is a ModuleList
            experts_is_a_list = (
                True if isinstance(layer_module.mlp.experts, torch.nn.modules.container.ModuleList) else False
            )
            # pr0(f"{experts_is_a_list=}", force=False)

            # performance: move specific params to cuda - so that all the tensor manipulation are done
            # on cuda - much much faster than on cpu
            #
            # move to cuda only the experts slice we will use on this rank
            orig_experts = layer_module.mlp.experts
            if experts_is_a_list:
                for i in local_expert_indices:
                    getattr(orig_experts[i], "gate_proj").to(device)
                    getattr(orig_experts[i], "up_proj").to(device)
                    getattr(orig_experts[i], "down_proj").to(device)
            if arctic_moe_config.use_shared_expert:
                orig_shared_expert = layer_module.mlp.shared_expert.to(device)
                orig_shared_expert_gate = layer_module.mlp.shared_expert_gate.to(device)

            see_memory_usage(f"{layer} after orig mlp to device", force=False)

            def copy_weights(from_name, to_param, local_expert_indices):
                if experts_is_a_list:  # ModuleList models like qwen
                    weight_stacked = torch.stack(
                        [getattr(orig_experts[i], from_name).weight.T for i in local_expert_indices]
                    )
                else:  # gpt-oss-like models with a stack of experts weights
                    weight_stacked = getattr(orig_experts, from_name)[local_expert_indices, ...]
                to_param.copy_(weight_stacked)

            # pr0(f"{local_expert_indices=}", force=True)
            # qwen -> unified gate_up interleaved on dim=-1 tensor like gpt-oss
            if experts_is_a_list:
                # pr0(f"{orig_experts[0].gate_proj.weight.shape=}", force=True)

                # 1. mlp.gate => router_gate
                arctic_moe.router_gate.copy_(layer_module.mlp.gate.weight)

                # 2. normal experts
                # 2a. gate_proj + up_proj => expert_gate_up
                # orig_experts[0].gate_proj.weight [hidden_size, intermediate_size]
                # gate_stacked.shape == [num_local_experts, intermediate_size, hidden_size]

                gate_stacked = torch.stack(
                    [getattr(orig_experts[i], "gate_proj").weight.T for i in local_expert_indices]
                )
                # pr0(f"{gate_stacked.shape=}", force=True)
                # same shape as gate_stacked
                up_stacked = torch.stack([getattr(orig_experts[i], "up_proj").weight.T for i in local_expert_indices])
                # pr0(f"{up_stacked.shape=}", force=True)

                # pr0(f"util 1 {tensor_has_nan(up_stacked)}", force=True)
                # putting the gate and up weigths in every-other order to match arctic-moe style

                gate_up = torch.stack((gate_stacked, up_stacked), dim=-1).view(*up_stacked.shape[:-1], -1).contiguous()
                # pr0(f"{gate_up.shape=}", force=True)
                # pr0(f"{arctic_moe.expert_gate_up.shape=}", force=True)
                arctic_moe.expert_gate_up.copy_(gate_up)

                # 2b. down_proj -> expert_down
                copy_weights("down_proj", arctic_moe.expert_down, local_expert_indices)

                # 3. shared expert
                if arctic_moe_config.use_shared_expert:
                    pr0("shared expert detected", force=True)

                    # a. gate_proj + up_proj -> shared_expert_gate_up
                    shared_expert_gate = orig_shared_expert.gate_proj.weight.T
                    shared_expert_up = orig_shared_expert.up_proj.weight.T
                    shared_expert_gate_up = (
                        torch.stack((shared_expert_gate, shared_expert_up), dim=-1)
                        .view(*shared_expert_up.shape[:-1], -1)
                        .contiguous()
                    )
                    arctic_moe.shared_expert_gate_up.copy_(shared_expert_gate_up)
                    pr0(f"{arctic_moe.shared_expert_gate_up.shape=}", force=True)

                    # b. down_proj -> shared_expert_down
                    shared_expert_down = orig_shared_expert.down_proj.weight.T
                    arctic_moe.shared_expert_down.copy_(shared_expert_down)

                    # c. shared_expert_gate -> shared_expert_output_gate
                    shared_expert_gate = orig_shared_expert_gate.weight.T
                    arctic_moe.shared_expert_output_gate.copy_(shared_expert_gate)

            else:  # gpt-oss

                # 1. mlp.router => router_gate
                arctic_moe.router_gate.copy_(layer_module.mlp.router.weight)
                # 2. gate_up_proj -> expert_gate_up
                copy_weights("gate_up_proj", arctic_moe.expert_gate_up, local_expert_indices)
                # 3. down_proj -> expert_down
                copy_weights("down_proj", arctic_moe.expert_down, local_expert_indices)

        if not is_resume:
            copy_weights()

        # pr0(f"util 3 {tensor_has_nan(arctic_moe.expert_gate_up)}", force=True)
        # pr0(f"util 4 {tensor_has_nan(arctic_moe.expert_down)}", force=True)

        see_memory_usage(f"{layer} before release", force=False)
        # layer_module.mlp.experts.to(meta_device)
        # layer_module.mlp.shared_expert.to(meta_device)
        # layer_module.mlp.shared_expert_gate.to(meta_device)
        layer_module.mlp.to(meta_device)

        # stash the original - to hide it the assignment value can be anything but nn.Module or nn.ModuleList
        arctic_moe._hide = dict(orig_mlp=layer_module.mlp)
        see_memory_usage(f"{layer} after release", force=False)

        # override the original with unified representation
        # 1. store the original structure for later restoration
        # layer_module.mlp_orig = layer_module.mlp.to(meta_device)
        # 2. now hijack it with our structure
        layer_module.mlp = arctic_moe

        duration = time.time() - start
        pr0(f"{layer_num}: duration {duration:.3f}secs", force=True)

        # pr0(f"{layer_module.mlp}", force=True)

    pr0(f"Rewritten model: {model}", force=True)


@torch.no_grad()
def remap_arctic_moe_to_orig_moe_mlp_params(model):
    """
    Undoes remap_orig_moe_mlp_params_to_arctic_moe, renaming the arctic_moe unified representation to the original model. The weights are gathered on rank 0 only.

    Args:
      - model: expects an unwrapped model object
    """
    device = model.device
    # meta_device = torch.device("meta")

    for layer_num, layer_module in enumerate(model.model.layers):
        # some models don't have moe in every layer
        if not isinstance(layer_module.mlp, ArcticMoE):
            # print(f"{layer_num} is not an MoE layer")
            continue
        print(f"{layer_num} is an MoE layer")
        arctic_moe = layer_module.mlp

        # if EP!=DP (i.e. MoE replicas) rank 0 could be different than `0`
        ep_group_rank_0 = dist.get_global_rank(arctic_moe.ep_group, 0)

        # leave the original intact in case this is an interim model saving and not an exit
        orig_mlp = copy.deepcopy(arctic_moe._hide["orig_mlp"])
        orig_mlp.to_empty(device=device)  # move out of meta

        # 1. router_gate => mlp.gate
        if arctic_moe.ep_rank == 0:
            orig_mlp.gate.weight.copy_(arctic_moe.router_gate)

        # 2. gate+up
        # gather gate_up from all ep ranks to rank 0
        if arctic_moe.ep_rank == 0:
            gate_up_list = [
                torch.zeros_like(arctic_moe.expert_gate_up, device=device) for i in range(arctic_moe.ep_size)
            ]
        else:
            gate_up_list = None

        dist.gather(arctic_moe.expert_gate_up, gate_up_list, dst=ep_group_rank_0, group=arctic_moe.ep_group)

        if arctic_moe.ep_rank == 0:
            gate_proj = []
            up_proj = []
            for gate_up in gate_up_list:
                gate_unstacked, up_unstacked = torch.unbind(gate_up.view(-1, 2), dim=-1)
                gate_unstacked = gate_unstacked.view(*gate_up.shape[:-1], -1)
                up_unstacked = up_unstacked.view(*gate_up.shape[:-1], -1)
                gate_proj += [x.T for x in gate_unstacked.unbind()]
                up_proj += [x.T for x in up_unstacked.unbind()]
            del gate_up_list  # free memory
            for i, expert in enumerate(orig_mlp.experts):
                expert.gate_proj.weight.copy_(gate_proj[i])
                expert.up_proj.weight.copy_(up_proj[i])

        # 3. down
        if arctic_moe.ep_rank == 0:
            down_list = [torch.zeros_like(arctic_moe.expert_down, device=device) for i in range(arctic_moe.ep_size)]
        else:
            down_list = None
        dist.gather(arctic_moe.expert_down, down_list, dst=ep_group_rank_0, group=arctic_moe.ep_group)

        if arctic_moe.ep_rank == 0:
            down_proj = []
            for down in down_list:
                down_proj += [x.T for x in down.unbind()]
            del down_list
            for i, expert in enumerate(orig_mlp.experts):
                expert.down_proj.weight.copy_(down_proj[i])

        # 4. shared expert
        if arctic_moe._config.use_shared_expert:

            orig_shared_expert = orig_mlp.shared_expert
            orig_shared_expert_gate = orig_mlp.shared_expert_gate

            # a. shared_expert_gate_up
            gate_up = arctic_moe.shared_expert_gate_up
            gate_unstacked, up_unstacked = torch.unbind(gate_up.view(-1, 2), dim=-1)
            gate_unstacked = gate_unstacked.view(*gate_up.shape[:-1], -1)
            up_unstacked = up_unstacked.view(*gate_up.shape[:-1], -1)
            orig_shared_expert.gate_proj.weight.copy_(gate_unstacked.T)
            orig_shared_expert.up_proj.weight.copy_(up_unstacked.T)

            # b. shared_expert_down
            orig_shared_expert.down_proj.weight.copy_(arctic_moe.shared_expert_down.T)

            # c. shared_expert_output_gate
            orig_shared_expert_gate.weight.copy_(arctic_moe.shared_expert_output_gate.T)

        # XXX: preserve and later restore if it's not an exit?
        # amoe_mlp = layer_module.mlp
        layer_module.mlp = orig_mlp


def identify_expert_params(model, ep_size):
    """
    This util:
    1. creates a list of data pointers to expert params, which is used by split_params_into_different_moe_groups_for_optimizer to split zero params from expert params in the optimizer param groups - we need to do it since we have no names in param groups.
    2. assigns 2 attributes: p.allreduce and p.group_name to expert params, which DS-ZeRO2 expects to idenfity if a param is an expert param and should be handled differently from ZeRO params.

    Call it before split_params_into_different_moe_groups_for_optimizer, which is called just before the optimizer is created.

    Args:
    - model: unwrapped model
    - ep_size: expert parallel size

    Returns:
    - expert_param_data_ptrs: a list of data pointers to expert params

    """

    # regex = r'arctic_moe.experts.deepspeed_experts.*?.weight'
    # regex = r"experts"
    # expert_param_data_ptrs = [p.data_ptr for n, p in model.named_parameters() if re.search(regex, n) is not None]

    expert_group_name = f"ep_size_{ep_size}"

    # router_gate + shared expert params aren't part of moe params for EP
    expert_param_names = ["expert_gate_up", "expert_down"]
    expert_param_data_ptrs = []
    for n, m in model.named_modules():
        # print(n)
        if isinstance(m, ArcticMoE):
            for n, p in m.named_parameters():
                if n.split(":")[-1] in expert_param_names:
                    expert_param_data_ptrs.append(p.data_ptr)
                    # XXX: DS-MoE legacy flags to integrate with ZeRO
                    # 1. this attribute is used in is_moe_param util https://github.com/deepspeedai/DeepSpeed/blob/69e03e52d0ebc567d34a163e925899751f7dbcb8/deepspeed/moe/utils.py#L28
                    p.allreduce = False
                    # 2. this attribute is used to discover which process group to use for reducing gradients https://github.com/deepspeedai/DeepSpeed/blob/69e03e52d0ebc567d34a163e925899751f7dbcb8/deepspeed/moe/utils.py#L116
                    p.group_name = expert_group_name

    print(f"Found {len(expert_param_data_ptrs)} MoE params")
    return expert_param_data_ptrs


def is_expert_param(param: torch.Tensor, expert_param_data_ptrs: List) -> bool:
    # XXX: this will be set once MoE class is used - find a better way to communicate it's an expert param
    if param.data_ptr in expert_param_data_ptrs and hasattr(param, "group_name"):
        # print(F"MoE param")
        return True
    else:
        return False


def split_params_into_different_moe_groups_for_optimizer(
    param_groups: Union[Dict[str, Any], Tuple[Dict[str, Any], ...], List[Dict[str, Any]]],
    expert_param_data_ptrs: List,
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

    # pr("Splitting into Moe")

    # gather all data parallel group names
    data_parallel_group_names: Set[str] = set()
    for param_group in param_groups:
        # for param in cast(List[nn.Parameter], param_group["params"]):
        for param in param_group["params"]:
            if is_expert_param(param, expert_param_data_ptrs):
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
            if is_expert_param(param, expert_param_data_ptrs):
                group_moe[param_group["name"]][param.group_name]["params"].append(param)
            else:
                new_params.append(param)
        param_group["params"] = new_params

    # XXX: the original code was splitting moe-groups into multiple subgroups of some random group size
    # https://github.com/deepspeedai/DeepSpeed/blob/69e03e52d0ebc567d34a163e925899751f7dbcb8/deepspeed/moe/utils.py#L121
    # it was added here https://github.com/deepspeedai/DeepSpeed/pull/2079
    # I removed it for now since I don't think it's relevant to non-fp16 use cases - can revisit when optimizing
    # Flatten the moe groups
    for moe_group in group_moe.values():
        for param_group in moe_group.values():
            param_groups.append(param_group)

    return param_groups


def get_expert_ckpt_name(save_dir, tag, ep_rank, moe_layer_id):
    ckpt_name = os.path.join(save_dir, tag, f"ep_rank_{ep_rank}_moe_layer_{moe_layer_id}_model_states.pt")
    return ckpt_name


def amoe_save_checkpoint(self, save_dir, tag, client_state={}, exclude_frozen_parameters=False):
    """
    It looks like we have to take care of saving everything in the model when using MoE and not just MoE params.
    """

    print(f"AMoE Checkpoint saving into {save_dir}/{tag}")
    save_path = self._get_ckpt_name(save_dir, tag)

    # XXX: for now ignoring expert_data_parallel

    ep_rank = 0
    moe_layer_id = 0
    for n_module, module in self.module.named_modules():

        if isinstance(module, ArcticMoE):
            ep_rank = module.ep_rank  # this will also be used later for optimizer states
            # get all moe parameters
            moe_state_dict = {}
            for n, p in module.state_dict().items():
                # if 'expert' in n and 'moe.gate.wg.weight' not in n:
                moe_state_dict[n_module + "." + n] = p
            # moe_str_prefix = ".arctic_moe."
            # pr0(f"saving keys {moe_state_dict.keys()}", force=True)
            # for k,v in moe_state_dict.items():
            #     print(f"{k} {v.shape=} {v.device}")
            see_memory_usage(f"{moe_layer_id} {n_module} after saving", force=True)

            moe_save_path = get_expert_ckpt_name(save_dir, tag, ep_rank, moe_layer_id)
            if self.checkpoint_engine.preserves_storage_sharing():
                saveable_state_dict = clone_tensors_for_torch_save(moe_state_dict)
            else:
                saveable_state_dict = moe_state_dict

            self.checkpoint_engine.save(saveable_state_dict, moe_save_path)

            moe_layer_id += 1

    self._curr_ckpt_path = os.path.join(save_dir, tag)

    # Save optimizer states. They are different across each EP rank
    optimizer_state = {
        "optimizer": self.optimizer.state_dict() if self.optimizer and not self.zero_optimization() else None
    }
    if self.checkpoint_engine.preserves_storage_sharing():
        saveable_state_dict = clone_tensors_for_torch_save(optimizer_state)
    else:
        saveable_state_dict = optimizer_state
    file_path = self._get_optimizer_ckpt_name(save_dir, tag, ep_rank)
    self.checkpoint_engine.save(saveable_state_dict, file_path)

    # Load flow uses below saved file for model parameters, RNG and more
    if ep_rank == 0:
        # Get non-moe parameters
        # Classes DeepSpeedEngine and PipelineEngine have different behavior for method module_state_dict.
        # DeepSpeedEngine returns the state dict, where PipelineEngine saves the state dict and returns None.
        # We need to get the state dict, therefore, call to DeepSpeedEngine (base class for PipelineEngine)
        full_state_dict = DeepSpeedEngine.module_state_dict(self, exclude_frozen_parameters=exclude_frozen_parameters)
        non_moe_state_dict = {k: v for k, v in full_state_dict.items() if not ("expert" in k or "router_gate" in k)}

        # TODO: update num experts info,.. in checkpoint
        state = dict(
            module=non_moe_state_dict,
            lr_scheduler=self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            data_sampler=(
                self.training_dataloader.data_sampler.state_dict()
                if (self.training_dataloader is not None and self.curriculum_learning_enabled())
                else None
            ),
            random_ltd=self.random_ltd_scheduler.state_dict() if self.random_ltd_enabled() else None,
            sparse_tensor_module_names=self.sparse_tensor_module_names,
            skipped_steps=self.skipped_steps,
            global_steps=self.global_steps,
            global_samples=self.global_samples,
            dp_world_size=self.dp_world_size,
            mp_world_size=self.mp_world_size,
            num_experts=self.num_experts,
        )
        state.update(client_state)
        pr0(f"Saving model checkpoint: {save_path}", force=True)
        if self.checkpoint_engine.preserves_storage_sharing():
            saveable_state_dict = clone_tensors_for_torch_save(state)
        else:
            saveable_state_dict = state
        self.checkpoint_engine.save(saveable_state_dict, save_path)

    # exit()


# XXX: additionally to perform a fast resume we also need to create a subclass with the original moe mlp blocks overridden with AMoE and its config somehow as well and use that in model/hf_factory.py -
#


def amoe_load_state_dict(
    checkpoint_path,
    tag,
    state_dict,
    old_moe_load,
    model=None,
    mpu=None,
    num_experts=1,
    checkpoint_engine=TorchCheckpointEngine(),
):
    print("AMoE Checkpoint loading")

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    see_memory_usage("before AMoE loading", force=True)
    moe_layer_id = 0
    for n_module, module in model.named_modules():
        pr0(f"{n_module}")
        if isinstance(module, ArcticMoE):
            ep_rank = module.ep_rank
            moe_ckpt_path = get_expert_ckpt_name(checkpoint_path, tag, ep_rank, moe_layer_id)
            ep_state_dict = checkpoint_engine.load(moe_ckpt_path, map_location=device)
            pr0(f"loading keys {ep_state_dict.keys()} from {moe_ckpt_path}", force=True)
            see_memory_usage(f"{moe_layer_id} {n_module} after loading", force=True)
            # for k,v in ep_state_dict.items():
            #     print(f"{k} {v.shape=} {v.device}")
            state_dict.update(ep_state_dict)
            moe_layer_id += 1
    see_memory_usage("after AMoE loading", force=True)
