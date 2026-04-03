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


class RaggedMoEGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inf_module,
        moe_output: torch.Tensor,
        scores: torch.Tensor,
        mapped_slots: torch.Tensor,
        expert_counts: torch.Tensor,
        normalize_scores: bool,
    ) -> torch.Tensor:
        n_tokens, _ = scores.shape[0], scores.shape[1]
        _, hidden_size = moe_output.shape
        layer_output = torch.empty(
            (n_tokens, hidden_size), device=moe_output.device, dtype=moe_output.dtype, requires_grad=True
        )
        inf_module.moe_gather(layer_output, moe_output, scores, mapped_slots, expert_counts, normalize_scores)
        ctx.save_for_backward(moe_output, scores, mapped_slots)
        ctx.normalize_scores = normalize_scores
        ctx.inf_module = inf_module
        return layer_output

    @staticmethod
    def backward(ctx, grad_layer_output: torch.Tensor):
        moe_output, scores, mapped_slots = ctx.saved_tensors
        grad_moe_output = torch.zeros_like(moe_output)
        grad_scores = torch.zeros_like(scores)
        ctx.inf_module.moe_gather_backward(
            grad_layer_output.contiguous(), grad_scores, grad_moe_output, moe_output, scores, mapped_slots
        )
        return None, grad_moe_output, grad_scores, None, None, None


class RaggedMoEGatherModule(torch.nn.Module):
    def __init__(self, normalize_scores: bool = False) -> None:
        super().__init__()
        self.normalize_scores = normalize_scores
        self.inf_module = RaggedOpsBuilder().load()

    def forward(
        self, moe_output: torch.Tensor, scores: torch.Tensor, mapped_slots: torch.Tensor, expert_counts: torch.Tensor
    ) -> torch.Tensor:
        return RaggedMoEGatherFunction.apply(
            self.inf_module, moe_output, scores, mapped_slots, expert_counts, self.normalize_scores
        )
