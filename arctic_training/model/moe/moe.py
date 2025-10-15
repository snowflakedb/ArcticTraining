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

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass(kw_only=True)
class MoEConfig:
    act_fn: object
    ep_group: dist.ProcessGroup
    ep_size: int
    input_dtype: torch.dtype
    intermediate_dim: int
    is_gated: bool = True
    loss_coeff: float = 0.01
    model_dim: int
    normalize_topk_scores: bool
    num_experts: int
    return_router_scores: bool
    top_k: int
    use_triton: bool = True


def torch_group_gemm_fn(A, B, rows_cumsum):
    C = torch.zeros((rows_cumsum[-1], B.shape[-1]), device=A.device, dtype=A.dtype)
    for i in range(len(rows_cumsum)):
        start = 0 if i == 0 else rows_cumsum[i - 1]
        end = rows_cumsum[i]
        C[start:end, :] = torch.matmul(A[start:end, :], B[i])
    return C


class ArcticMoE(nn.Module):
    """Mixture of Experts (MoE) layer.
    Args:
        config: MoEConfig object
    """

    def __init__(self, config: MoEConfig):
        super(ArcticMoE, self).__init__()
        self._config = config

        self.act_fn = config.act_fn
        self.ep_group = config.ep_group
        self.ep_size = config.ep_size
        self.input_dtype = config.input_dtype
        self.intermediate_dim = config.intermediate_dim
        self.model_dim = config.model_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        self.num_local_experts = self.num_experts // self.ep_size

        # XXX: shouldn't be inside expert param group - should be data parallel
        self.router_gate = nn.Parameter(
            torch.empty(
                (self.num_experts, self.model_dim),
                dtype=self.input_dtype,
            )
        )

        # Initialize expert weights
        self.expert_gate_up = nn.Parameter(
            torch.empty(
                (
                    self.num_local_experts,
                    self.model_dim,
                    (2 * self.intermediate_dim) if config.is_gated else self.intermediate_dim,
                ),
                dtype=self.input_dtype,
            )
        )

        self.gate_scale = 1.0
        self.up_scale = 0.0

        self.expert_down = nn.Parameter(
            torch.empty((self.num_local_experts, self.intermediate_dim, self.model_dim), dtype=self.input_dtype)
        )

        self.comm_stream = torch.cuda.Stream()
        if config.use_triton:
            from arctic_training.model.moe.moe_gemm import group_gemm_fn

            self.moegemm = group_gemm_fn
        else:
            self.moegemm = torch_group_gemm_fn

    def GroupGeMM(self, x, expert_token_count_cumsum):
        """Grouped GEMM for MoE experts.
        Args:
            x: Input tensor of shape [#tokens * topk, model_dim]
            expert_token_count_cumsum: ???
        Returns:
            Output tensor after applying expert weights and activation.
        """
        n_topk_tokens = x.shape[0]
        output = torch.empty_like(x)
        intermediate = torch.empty((n_topk_tokens, self.intermediate_dim), dtype=self.input_dtype, device=x.device)
        intermediate = self.moegemm(x, self.expert_gate_up, expert_token_count_cumsum)
        if self._config.is_gated:
            gate, up = intermediate[..., 0::2], intermediate[..., 1::2]
            gate = self.act_fn(gate * self.gate_scale)
            intermediate = (up + self.up_scale) * gate
        output = self.moegemm(intermediate, self.expert_down, expert_token_count_cumsum)
        return output

    def local_ep_transpose(self, x, expert_token_rcv_count):
        x_split = torch.split(x, expert_token_rcv_count.tolist(), dim=0)
        x = torch.cat(
            [
                x_split[i * self.num_local_experts + j]
                for j in range(self.num_local_experts)
                for i in range(self.ep_size)
            ],
            dim=0,
        )
        expert_token_count_cumsum = expert_token_rcv_count.view(self.ep_size, self.num_local_experts).sum(0).cumsum(0)
        expert_token_count_transposed = (
            expert_token_rcv_count.view(self.ep_size, self.num_local_experts).t().reshape(-1)
        )
        return x, expert_token_count_cumsum, expert_token_count_transposed

    def local_ep_depermute(self, x, expert_token_count_transposed):
        out_split = torch.split(x, expert_token_count_transposed.tolist(), dim=0)
        x = torch.cat(
            [out_split[i * self.ep_size + j] for j in range(self.ep_size) for i in range(self.num_local_experts)],
            dim=0,
        )
        return x

    def forward(self, hidden_states):
        # Forward pass through the MoE layer
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states, self.router_gate)
        moe_input, expert_token_count, expert_token_rcv_count, scores, token_mapped_slots = self.MoERouter(
            hidden_states, logits
        )

        if self.ep_size > 1:
            moe_input = self.AlltoAllV(moe_input, expert_token_count, expert_token_rcv_count)
            moe_input, expert_token_count_cumsum, expert_token_count_transposed = self.local_ep_transpose(
                moe_input, expert_token_rcv_count
            )
        else:
            expert_token_count_cumsum = expert_token_count.cumsum(0)
        moe_output = self.GroupGeMM(moe_input, expert_token_count_cumsum)

        if self.ep_size > 1:
            moe_output = self.local_ep_depermute(moe_output, expert_token_count_transposed)
            moe_output = self.AlltoAllV(moe_output, expert_token_rcv_count, expert_token_count)

        output = self.MoECombine(moe_output, token_mapped_slots, scores)
        output = output.reshape(orig_shape)

        return (output, scores) if self._config.return_router_scores else output

    def AlltoAllV(self, x, token_snd_count=None, token_rcv_count=None):
        """AlltoAllV operation for distributed MoE.
        Args:
            x: Input tensor
            token_snd_count: Number of elements to send to each rank
            token_rcv_count: Number of elements to receive from each rank
        Returns:
            Output tensor after AlltoAllV
        """
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        token_snd_count = token_snd_count.reshape(self.ep_size, -1).sum(dim=-1)
        token_rcv_count = token_rcv_count.reshape(self.ep_size, -1).sum(dim=-1)
        input_splits = token_snd_count.tolist()
        output_splits = token_rcv_count.tolist()
        output = torch.empty((sum(output_splits), x.shape[1]), dtype=x.dtype, device=x.device)
        dist.all_to_all_single(
            output, x, output_split_sizes=output_splits, input_split_sizes=input_splits, group=self.ep_group
        )
        return output

    def MoERouter(self, hidden_states, logits):
        """Mixture of Experts (MoE) router.
        Args:
            hidden_states: [#tokens, hidden_size]
            logits: [#tokens, num_experts]
        Returns:
            moe_input: [#tokens * topk, hidden_size]
            expert_token_count: [num_experts]
            expert_token_rcv_count: [num_experts]
            scores: [#tokens, topk]
            token_mapped_slots: [#tokens * topk]
        """
        scores, token_mapped_slots, expert_token_count, expert_token_rcv_count = self._gate(logits)
        moe_input = hidden_states[token_mapped_slots]
        return moe_input, expert_token_count, expert_token_rcv_count, scores, token_mapped_slots

    def _gate(self, logits):
        logits = logits.view(-1, self.num_experts)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)  # T x E

        topk_scores, topk_expert_indices = torch.topk(probs, k=self.top_k, dim=-1)  # T x top_k
        if self._config.normalize_topk_scores:
            topk_scores = topk_scores / torch.sum(topk_scores, dim=-1, keepdim=True)
        topk_scores = topk_scores.to(self._config.input_dtype)  # T x top_k
        topk_scores = topk_scores.t().reshape(-1)
        topk_expert_indices = topk_expert_indices.t().reshape(-1)

        T = probs.shape[0]

        topk_expert_mask = F.one_hot(topk_expert_indices, num_classes=self.num_experts).t()  # E x (T * top_k)

        token_mapped_slots = torch.cat([torch.where(topk_expert_mask[i])[0] for i in range(topk_expert_mask.shape[0])])
        topk_scores = topk_scores[token_mapped_slots]
        token_mapped_slots = token_mapped_slots % T

        expert_token_count = topk_expert_mask.sum(dim=-1)  # E
        total_count = expert_token_count.sum()
        expert_freq = expert_token_count / total_count

        if self.ep_size > 1:
            expert_token_rcv_count = torch.empty_like(expert_token_count)
            with torch.cuda.stream(self.comm_stream):
                dist.all_to_all_single(expert_token_rcv_count, expert_token_count, group=self.ep_group)
        else:
            expert_token_rcv_count = expert_token_count

        def _load_balance_grad_hook(grad):
            """
            We don't explicitly collect the LB loss. Instead, we analytically compute the grad of the
            LB loss wrt `probs` and use that to modify the original grad via grad hook.

            - LB loss is defined as `sum(prob * freq)` where `prob = mean(probs, dim=0)`.
            - grad of the LB loss wrt `probs` is therefore `freq / probs.shape[0]`.
            - grad (hence the corresponding LB loss) is further adjusted by `num_experts`, `num_layers`
              and `num_microbatches` to ensure the invariance of loss magnitude across model settings.
            """
            # get_num_microbatches() TODO(Reza): get the actual number of microbatches
            coeff = self._config.loss_coeff / T / 1
            return grad + expert_freq.unsqueeze(0) * coeff / self.ep_size

        if probs.requires_grad:
            probs.register_hook(_load_balance_grad_hook)
        # print(f"router logits norm {topk_scores.norm().item():.3f}")
        return topk_scores, token_mapped_slots, expert_token_count, expert_token_rcv_count

    def MoECombine(self, moe_output, token_mapped_slots, scores):
        """MoE gather operation.
        Args:
            moe_output: [#tokens * topk, hidden_size]
        Returns:
            output: [#tokens, hidden_size]
        """
        output = torch.zeros(
            (moe_output.shape[0] // self.top_k, self.model_dim), dtype=moe_output.dtype, device=moe_output.device
        )
        output.index_add_(0, token_mapped_slots, moe_output * scores[:, None])
        return output
