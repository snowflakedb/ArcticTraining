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

from arctic_training.model.moe.moe_gemm import group_gemm_fn
from arctic_training.model.moe.moe_gemm import torch_group_gemm_fn


@dataclass
class MoEConfig:
    num_experts: int
    model_dim: int
    intermediate_dim: int
    top_k: int
    input_dtype: torch.dtype
    activation: str
    normalize_scores: bool
    is_gated: bool = True
    loss_coeff: float = 0.01
    use_triton: bool = True


class ArcticMoE(nn.Module):
    """Mixture of Experts (MoE) layer.
    Args:
        config: MoEConfig object
    """

    def __init__(self, config: MoEConfig, ep_group):
        super(ArcticMoE, self).__init__()
        self._config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.model_dim = config.model_dim
        self.intermediate_dim = config.intermediate_dim
        self.input_dtype = config.input_dtype

        if config.activation == "relu":
            self._activation = F.relu
        elif config.activation == "gelu":
            self._activation = F.gelu
        elif config.activation == "silu":
            self._activation = F.silu
        elif config.activation is None:
            self._activation = None
        else:
            raise ValueError(f"Unsupported activation {config.activation}")

        self._gate_proj = nn.Linear(self.model_dim, self.num_experts, bias=False).to(self.input_dtype)
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(group=self.ep_group)
        num_local_experts = self.num_experts // self.ep_size

        # Initialize expert weights
        self.expert_gate_up_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                self.model_dim,
                (2 * self.intermediate_dim) if config.is_gated else self.intermediate_dim,
                dtype=self.input_dtype,
            )
        )
        self.gate_scale = 1.0
        self.up_scale = 0.0

        self.expert_down_weight = nn.Parameter(
            torch.empty(num_local_experts, self.intermediate_dim, self.model_dim, dtype=self.input_dtype)
        )

        self.comm_stream = torch.cuda.Stream()
        self.moegemm = group_gemm_fn if config.use_triton else torch_group_gemm_fn

    def GroupGeMM(self, x, expert_token_rcv_count):
        """Grouped GEMM for MoE experts.
        Args:
            x: Input tensor of shape [#tokens * topk, model_dim]
        Returns:
            Output tensor after applying expert weights and activation.
        """
        n_topk_tokens = x.size(0)
        output = torch.empty_like(x)
        intermediate = torch.empty((n_topk_tokens, self.intermediate_dim), dtype=self.input_dtype, device=x.device)
        # TODO(Reza): we need to add a transformation kernel that put the local-experts together!
        expert_count_cumsum = expert_token_rcv_count.view(-1, self.ep_size).sum(-1).cumsum(0)
        intermediate = self.moegemm(x, self.expert_gate_up_weight, expert_count_cumsum)
        if self._config.is_gated:
            gate, up = intermediate[0::2], intermediate[1::2]
            gate = self._activation(gate * self.gate_scale) if self._activation else intermediate
            intermediate = (up + self.up_scale) * up
        output = self.moegemm(intermediate, self.expert_down_weight, expert_count_cumsum)
        return output

    def forward(self, hidden_states):
        # Forward pass through the MoE layer
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = self._gate_proj(hidden_states)
        (moe_input, expert_token_count, expert_token_rcv_count, scores, token_mapped_slots) = self.MoERouter(
            hidden_states, logits
        )
        moe_input = self.AlltoAllV(moe_input, expert_token_count, expert_token_rcv_count)
        moe_output = self.GroupGeMM(moe_input, expert_token_rcv_count)
        moe_output = self.AlltoAllV(moe_output, expert_token_rcv_count, expert_token_count)
        output = self.MoECombine(moe_output, token_mapped_slots)
        output = output.reshape(orig_shape)
        return output

    def AlltoAllV(self, x, token_snd_count=None, token_rcv_count=None):
        """AlltoAllV operation for distributed MoE.
        Args:
            x: Input tensor
            token_snd_count: Counts of elements to snd to each rank
            token_rcv_count: Counts of elements to receive from each rank
        Returns:
            Output tensor after AlltoAllV
        """
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        token_snd_count = token_snd_count.reshape(self.ep_size, -1).sum(dim=-1)
        token_rcv_count = token_rcv_count.reshape(self.ep_size, -1).sum(dim=-1)
        input_splits = token_snd_count.tolist()
        output_splits = token_rcv_count.tolist()
        output = torch.empty((sum(output_splits), x.size(1)), dtype=x.dtype, device=x.device)
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

        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)  # T x E

        _, topk_expert_indices = torch.topk(probs, k=self.top_k, dim=-1)  # T x top_k
        topk_scores = torch.gather(probs, dim=-1, index=topk_expert_indices).to(self._config.input_dtype)  # T x top_k
        topk_scores = topk_scores.t().reshape(-1)
        topk_expert_indices = topk_expert_indices.t().reshape(-1)

        T = probs.size(0)

        topk_expert_mask = F.one_hot(topk_expert_indices, num_classes=self.num_experts).t()  # E x (T * top_k)

        token_mapped_slots = torch.cat(
            [torch.where(topk_expert_mask[i])[0] % T for i in range(topk_expert_mask.shape[0])]
        )
        topk_scores = topk_scores[token_mapped_slots]

        expert_token_count = topk_expert_mask.sum(dim=-1)  # E
        total_count = expert_token_count.sum()
        expert_freq = expert_token_count / total_count

        expert_token_rcv_count = torch.empty_like(expert_token_count)
        with torch.cuda.stream(self.comm_stream):
            dist.all_to_all_single(expert_token_rcv_count, expert_token_count, group=self.ep_group)

        def _load_balance_grad_hook(grad):
            """
            We don't explicitly collect the LB loss. Instead, we analytically compute the grad of the
            LB loss wrt `probs` and use that to modify the original grad via grad hook.

            - LB loss is defined as `sum(prob * freq)` where `prob = mean(probs, dim=0)`.
            - grad of the LB loss wrt `probs` is therefore `freq / probs.size(0)`.
            - grad (hence the corresponding LB loss) is further adjusted by `num_experts`, `num_layers`
              and `num_microbatches` to ensure the invariance of loss magnitude across model settings.
            """
            coeff = (
                self._config.loss_coeff / T / 1
            )  # get_num_microbatches() TODO(Reza): get the actual number of microbatches
            return grad + expert_freq.unsqueeze(0) * coeff / self.ep_size

        if probs.requires_grad:
            probs.register_hook(_load_balance_grad_hook)
        return topk_scores, token_mapped_slots, expert_token_count, expert_token_rcv_count

    def MoECombine(self, moe_output, token_mapped_slots):
        """MoE gather operation.
        Args:
            moe_output: [#tokens * topk, hidden_size]
        Returns:
            output: [#tokens, hidden_size]
        """
        output = torch.empty(
            (moe_output.size(0) // self.top_k, self.model_dim), dtype=moe_output.dtype, device=moe_output.device
        )
        output.index_add_(0, token_mapped_slots, moe_output)
        return output
