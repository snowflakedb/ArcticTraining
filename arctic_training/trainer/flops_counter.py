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

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from https://github.com/volcengine/verl/blob/main/verl/utils/flops_counter.py

from transformers import PretrainedConfig

VALID_CONFIG_TYPE = {
    "llama",
    "qwen2",
    "qwen2_moe",
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3",
    "qwen3_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    "deepseek_v3",
    "minicpmv",
    "minicpmo",
    "mistral",
    "gemma3_text",
    "seed_oss",
    "apertus",
    "glm4v",
}


def estimate_decoder_transformer_tflos(
    hf_model_config, model_size, batch_of_seqlens, enable_gradient_checkpointing, forward_only=False
):
    """
    Tries to use a model-specific flop counter (adapted from verl) if such is available, otherwise falls back onto normal dense decoder flop counter.

    Args:
        - `hf_model_config`: HF model config object
        - `model_size`: total number of params
        - `batch_of_seqlens` is a bs-size list of lists, where each sub-list is seqlens of each sub-sample, or a single seqlen if these are unpacked samples.
        - `enable_gradient_checkpointing`: if grad checkpointing is enabled
        - `forward_only`: calculate only forward flops
    Returns:
        - tflos total
        - tokens total

    Examples of `batch_of_seqlens`:
    - bs=1 + packed samples:   [[100, 200, 4090]]
    - bs=2 + packed samples:   [[100, 200, 4090], [4090, 100, 200]]
    - bs=1 + an unpacked sample: [[4090]]
    - bs=2 + unpacked samples: [[4090], [4090]]
    """
    flops_counter = FlopsCounter(hf_model_config, model_size, enable_gradient_checkpointing)

    seqlen = 0
    tflos = 0
    # iterate over batch size
    for seqlens in batch_of_seqlens:
        tflos += flops_counter.estimate_tflos(batch_seqlens=seqlens, delta_time=1)
        seqlen += sum(seqlens)

    return tflos, seqlen


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig, model_size, enable_gradient_checkpointing=False):
        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_qwen2_flops,
            "qwen2_5_vl": self._estimate_qwen2_flops,
            "qwen3": self._estimate_qwen2_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "minicpmv": self._estimate_qwen2_flops,
            "minicpmo": self._estimate_qwen2_flops,
        }
        # fallback is self._estimate_dense_decoder_transformer_tflos

        self.config = getattr(config, "text_config", config)
        self.model_size = model_size

        if enable_gradient_checkpointing:
            self.dense_flops_multiplier = 8
            self.attn_flops_multiplier = 16
        else:
            self.dense_flops_multiplier = 6
            self.attn_flops_multiplier = 12

    def _dense_flops_multiplier(self, forward_only):
        return 2 if forward_only else self.dense_flops_multiplier

    def _attn_flops_multiplier(self, forward_only):
        return 2 if forward_only else self.attn_flops_multiplier

    def _estimate_dense_decoder_transformer_tflos(self, tokens_sum, batch_seqlens, delta_time=1, forward_only=False):
        """Given a sequence length, estimates the number of floating point operations required to run the model."""

        tokens_sum = 1  # noqa not used
        delta_time = 1  # noqa not used

        def _inner(config, model_size, seq_len):
            return (
                self._dense_flops_multiplier(forward_only) * model_size * seq_len
                + self._attn_flops_multiplier(forward_only)
                * config.num_hidden_layers
                * config.hidden_size
                * seq_len**2
            ) / 1e12

        tflos = sum(_inner(self.config, self.model_size, seqlen) for seqlen in batch_seqlens)

        return tflos

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        intermediate_size = self.config.intermediate_size

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = self._dense_flops_multiplier(forward_only) * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = (
            self._attn_flops_multiplier(forward_only)
            * seqlen_square_sum
            * head_dim
            * num_attention_heads
            * num_hidden_layers
        )

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        moe_intermediate_size = self.config.moe_intermediate_size
        num_hidden_layers = self.config.num_hidden_layers
        first_k_dense_replace = self.config.first_k_dense_replace
        num_query_heads = self.config.num_attention_heads
        moe_num_expert = self.config.n_routed_experts

        moe_topk = self.config.num_experts_per_tok
        share_expert_num = self.config.n_shared_experts

        # non-attn per layer parm
        moe_gata_N = hidden_size * moe_num_expert
        # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
        moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
        # MLA attn
        attn_linear_N = 0
        q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim
        if self.config.q_lora_rank is None:
            attn_linear_N += hidden_size * num_query_heads * q_head_dim
        else:
            attn_linear_N += hidden_size * self.config.q_lora_rank
            attn_linear_N += num_query_heads * q_head_dim * self.config.q_lora_rank

        attn_linear_N += hidden_size * (self.config.kv_lora_rank + self.config.qk_rope_head_dim)
        attn_linear_N += (
            num_query_heads
            * (q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim)
            * self.config.kv_lora_rank
        )
        attn_linear_N += num_query_heads * self.config.v_head_dim * hidden_size
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        moe_N = (
            (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
            + (hidden_size * self.config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
            + emd_and_lm_head_N
        )
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = self._dense_flops_multiplier(forward_only) * moe_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen * num_hidden_layers

        attn_qkv_flops = self._attn_flops_multiplier(forward_only) * seqlen_square_sum * q_head_dim * num_query_heads
        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12

        return flops_achieved

    def _estimate_qwen2_moe_flops(self, tokens_sum, batch_seqlens, delta_time, forward_only):
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        num_hidden_layers = self.config.num_hidden_layers
        num_key_value_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        moe_intermediate_size = self.config.moe_intermediate_size
        moe_topk = self.config.num_experts_per_tok
        num_experts = self.config.num_experts

        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_size = num_attention_heads * head_dim
        k_size = num_key_value_heads * head_dim
        v_size = num_key_value_heads * head_dim

        # non-attn per layer parm
        # gate + moe export
        moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = self._dense_flops_multiplier(forward_only) * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = (
            self._attn_flops_multiplier(forward_only)
            * seqlen_square_sum
            * head_dim
            * num_attention_heads
            * num_hidden_layers
        )

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_tflos(self, batch_seqlens, delta_time=1, forward_only=False):
        """
        Estimate the floating point op count based on the number of valid tokens in the current batch.
        Args:
            - batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the current batch.
            - delta_time (float): time input is ignored, as we just want the flos and not flops - it's here to keep up with the verl-implementation so it'd be easy to update in the future
            forward_only (bool): Compute forward pass flops only. Used for rollout.

        Returns:
            - estimated_flos (float): The estimated flos based on the input tokens and time.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_dense_decoder_transformer_tflos)
        estimated_tflops = func(tokens_sum, batch_seqlens, delta_time=1, forward_only=forward_only)
        return estimated_tflops
