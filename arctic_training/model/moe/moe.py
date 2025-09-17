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

from arctic_training.model.moe.mo_gemm import group_gemm_fn
from arctic_training.model.moe.mo_gemm import torch_group_gemm_fn


@dataclass
class MoEConfig:
    num_experts: int
    model_dim: int
    intermediate_dim: int
    top_k: int
    input_dtype: torch.dtype
    activation: str
    normalize_scores: bool
    loss_coeff: float = 0.01
    use_triton: bool = True


class ArcticMoE(nn.Module):
    """Mixture of Experts (MoE) layer.
    Args:
        config: MoEConfig object
    """

    def __init__(self, config: MoEConfig, ep_group=None):
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

        # Initialize expert weights
        self.expert_intermediate_weights = nn.Parameter(
            torch.empty(self.num_experts, self.model_dim, self.intermediate_dim, dtype=self.input_dtype)
        )
        self.expert_output_weights = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_dim, self.model_dim, dtype=self.input_dtype)
        )

        self.comm_stream = torch.cuda.Stream()
        self.moegemm = group_gemm_fn if config.use_triton else torch_group_gemm_fn

    def GroupGeMM(self, x):
        """Grouped GEMM for MoE experts.
        Args:
            x: Input tensor of shape [#tokens * topk, model_dim]
        Returns:
            Output tensor after applying expert weights and activation.
        """
        n_tokens_topk = x.size(0)
        output = torch.empty_like(x)
        intermediate = torch.empty((n_tokens_topk, self.intermediate_dim), dtype=self.input_dtype, device=x.device)
        expert_count_cumsum = self.rcv_expert_counts.cumsum(0)
        intermediate = self.moegemm(x, self.expert_intermediate_weights, expert_count_cumsum)
        intermediate = self._activation(intermediate) if self._activation else intermediate
        output = self.moegemm(intermediate, self.expert_output_weights, expert_count_cumsum)
        return output

    def forward(self, hidden_states):
        # Forward pass through the MoE layer
        logits = self._gate_proj(hidden_states)
        moe_input, expert_counts, rcv_expert_counts, scores, mapped_slots, expert_cumsum = self.MoERouter(
            hidden_states, logits
        )
        moe_input = self.AlltoAllV(moe_input, self.expert_counts, self.rcv_expert_counts)
        moe_output = self.GroupGeMM(moe_input)
        moe_output = self.AlltoAllV(moe_output, self.rcv_expert_counts, self.expert_counts)
        output = self.MoECombine(moe_output)
        return (output, expert_counts, scores, mapped_slots, expert_cumsum)

    def AlltoAllV(self, x, send_counts=None, rcv_counts=None):
        """AlltoAllV operation for distributed MoE.
        Args:
            x: Input tensor
            send_counts: Counts of elements to send to each rank
            rcv_counts: Counts of elements to receive from each rank
            group: Process group for communication
        Returns:
            Output tensor after AlltoAllV
        """
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        send_counts = send_counts.reshape(-1, self.ep_size).sum(dim=-1)
        rcv_counts = rcv_counts.reshape(-1, self.ep_size).sum(dim=-1)
        input_splits = send_counts.tolist()
        output_splits = rcv_counts.tolist()
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
            expert_counts: [num_experts]
            scores: [#tokens, topk]
        """
        scores, mapped_slots, expert_counts, rcv_counts = self._gate(logits)
        moe_input = hidden_states[mapped_slots]
        expert_cumsum = expert_counts.cumsum(0)
        return moe_input, expert_counts, rcv_counts, scores, mapped_slots, expert_cumsum

    def _gate(self, logits):
        logits = logits.view(-1, self.num_experts)

        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)  # T x E

        _, ind = torch.topk(probs, k=self.top_k, dim=-1)  # T x top_k
        val = torch.gather(probs, dim=-1, index=ind).to(self._config.input_dtype)  # T x top_k
        val = val.t().reshape(-1)
        ind = ind.t().reshape(-1)

        T = probs.size(0)

        weight = val[ind]
        mask = F.one_hot(ind, num_classes=self.num_experts).t()  # E x (T * top_k)

        count = mask.sum(dim=-1)  # E
        total = count.sum()
        freq = count / total

        rcv_count = torch.empty_like(count)
        with torch.cuda.stream(self.comm_stream):
            dist.all_to_all_single(rcv_count, count, group=self.ep_group)

        def _load_balance_grad_hook(grad):
            """
            We don't explicitly collect the LB loss. Instead, we analytically compute the grad of the
            LB loss wrt `probs` and use that to modify the original grad via grad hook.

            - LB loss is defined as `sum(prob * freq)` where `prob = mean(probs, dim=0)`.
            - grad of the LB loss wrt `probs` is therefore `freq / probs.size(0)`.
            - grad (hence the corresponding LB loss) is further adjusted by `num_experts`, `num_layers`
              and `num_microbatches` to ensure the invariance of loss magnitude across model settings.
            """
            if self.micro_batch_averaging == "tokenwise_sum_no_rescale":
                coeff = (
                    self._config.loss_coeff / 1
                )  # get_num_microbatches() TODO(Reza): get the actual number of microbatches
            else:
                coeff = (
                    self._config.loss_coeff / T / 1
                )  # get_num_microbatches() TODO(Reza): get the actual number of microbatches
            return grad + freq.unsqueeze(0) * coeff / self.ep_size

        if probs.requires_grad:
            probs.register_hook(_load_balance_grad_hook)
        return weight, ind, count, rcv_count

    def MoECombine(self, moe_output, mapped_slots):
        """MoE gather operation.
        Args:
            moe_output: [#tokens * topk, hidden_size]
        Returns:
            output: [#tokens, hidden_size]
        """
        output = torch.empty(
            (moe_output.size(0) // self.top_k, self.model_dim), dtype=moe_output.dtype, device=moe_output.device
        )
        output.index_add_(0, mapped_slots, moe_output[:: self.top_k])
        return output
