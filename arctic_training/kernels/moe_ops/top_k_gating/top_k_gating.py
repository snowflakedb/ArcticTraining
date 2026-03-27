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


class RaggedTopKGatingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inf_module, expert_counts, assignments, offsets, logits, replay_routing):

        ctx.save_for_backward(assignments, logits)
        ctx.inf_module = inf_module
        n_tokens = logits.size(0)
        top_k = assignments.size(1)
        scores = torch.empty(n_tokens, top_k, device=logits.device, dtype=logits.dtype, requires_grad=True)
        logits_out = torch.empty_like(logits)
        if replay_routing:
            inf_module.top_k_gating_with_replay(expert_counts, scores, assignments, offsets, logits, logits_out)
        else:
            inf_module.top_k_gating(expert_counts, scores, assignments, offsets, logits, logits_out)
        return expert_counts, scores, assignments, offsets, logits_out

    @staticmethod
    def backward(ctx, *grad_outputs):
        import pdb

        pdb.set_trace()
        assignments, logits = ctx.saved_tensors
        _, grad_scores, _, _, grad_logits = grad_outputs
        grad_scores = grad_scores.contiguous()
        grad_logits = grad_logits.contiguous()
        ctx.inf_module.top_k_gating_bwd(grad_logits, grad_scores, logits, assignments)
        return None, None, None, None, grad_logits


class RaggedTopKGatingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inf_module = RaggedOpsBuilder().load()

    def forward(self, expert_counts, assignments, offsets, logits, replay_routing=False):
        # import pdb; pdb.set_trace()
        return RaggedTopKGatingFunction.apply(
            self.inf_module, expert_counts, assignments, offsets, logits, replay_routing
        )
