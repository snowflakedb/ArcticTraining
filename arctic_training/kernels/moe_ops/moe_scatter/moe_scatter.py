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

import torch

from arctic_training.op_builder import RaggedOpsBuilder


class RaggedMoEScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inf_module, expert_cumsum, mapped_slots, activations, expert_counts, assignments, offsets):
        ctx.inf_module = inf_module
        ctx.n_experts = expert_counts.shape[0]
        ctx.save_for_backward(mapped_slots, assignments)
        n_tokens, n_top_k = assignments.shape
        num_experts = expert_counts.shape[0]
        max_capacity_per_expert = expert_counts.max()
        torch.distributed.all_reduce(max_capacity_per_expert, op=torch.distributed.ReduceOp.MAX)
        moe_input = torch.empty(
            max_capacity_per_expert * num_experts,
            activations.shape[1],
            device=activations.device,
            dtype=activations.dtype,
            requires_grad=True,
        )
        inf_module.moe_scatter(
            moe_input,
            expert_cumsum,
            mapped_slots,
            activations,
            expert_counts,
            assignments,
            offsets,
            max_capacity_per_expert,
        )
        return moe_input, expert_cumsum, mapped_slots, max_capacity_per_expert

    @staticmethod
    def backward(ctx, *grad_output):
        mapped_slots, assignments = ctx.saved_tensors
        grad_moe_input = grad_output[0].contiguous()
        n_tokens, n_top_k = assignments.shape

        grad_activations = torch.empty(
            n_tokens, grad_moe_input.shape[1], device=assignments.device, dtype=grad_moe_input.dtype
        )
        ctx.inf_module.moe_scatter_backward(grad_activations, grad_moe_input, mapped_slots, assignments, ctx.n_experts)
        return None, None, None, grad_activations, None, None, None


class RaggedMoEScatterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inf_module = RaggedOpsBuilder().load()

    def forward(self, expert_cumsum, mapped_slots, activations, expert_counts, assignments, offsets):
        return RaggedMoEScatterFunction.apply(
            self.inf_module, expert_cumsum, mapped_slots, activations, expert_counts, assignments, offsets
        )
