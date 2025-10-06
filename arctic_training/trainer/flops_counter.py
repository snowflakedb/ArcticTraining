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


def estimate_decoder_transformer_tflos(hf_model_config, model_size, batch_of_seqlens):
    """
    Tries to use a model-specific flop counter (adapted from verl) if such is available, otherwise falls back onto normal dense decoder flop counter.

    Args:
        - `hf_model_config`: HF model config object
        - `model_size`: total number of params
        - `batch_of_seqlens` is a bs-size list of lists, where each sub-list is seqlens of each sub-sample, or a single seqlen if these are unpacked samples.
    Returns:
        - tflos total
        - tokens total

    Examples of `batch_of_seqlens`:
    - bs=1 + packed samples:   [[100, 200, 4090]]
    - bs=2 + packed samples:   [[100, 200, 4090], [4090, 100, 200]]
    - bs=1 + an unpacked sample: [[4090]]
    - bs=2 + unpacked samples: [[4090], [4090]]
    """
    flops_counter = FlopsCounter(hf_model_config, model_size)

    seqlen = 0
    tflos = 0
    # iterate over batch size
    for seqlens in batch_of_seqlens:
        tflos += flops_counter.estimate_flops(batch_seqlens=seqlens, delta_time=1)
        seqlen += sum(seqlens)

    return tflos, seqlen


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig, model_size):
        self.estimate_func = {
            "qwen2": self._estimate_qwen2_flops,
            "llama": self._estimate_qwen2_flops,
            "qwen2_moe": self._estimate_qwen2_moe_flops,
            "qwen2_vl": self._estimate_qwen2_flops,
            "qwen2_5_vl": self._estimate_qwen2_flops,
            "qwen3": self._estimate_qwen2_flops,
            "qwen3_moe": self._estimate_qwen2_moe_flops,
            "qwen3_vl": self._estimate_qwen2_flops,
            "qwen3_vl_moe": self._estimate_qwen2_moe_flops,
            "deepseek_v3": self._estimate_deepseek_v3_flops,
            "minicpmv": self._estimate_qwen2_flops,
            "minicpmo": self._estimate_qwen2_flops,
            "mistral": self._estimate_qwen2_flops,
            "gemma3_text": self._estimate_gemma3_flops,
            "seed_oss": self._estimate_qwen2_flops,
            "apertus": self._estimate_apertus_flops,
            "glm4v": self._estimate_qwen2_flops,
        }
        # fallback is self._estimate_dense_decoder_transformer_tflos

        self.config = getattr(config, "text_config", config)
        self.model_size = model_size

    def _estimate_dense_decoder_transformer_tflos(self, tokens_sum, batch_seqlens, delta_time=1):
        """Given a sequence length, estimates the number of floating point operations required to run the model.
        It currently hardwires activation checkpointing always on (co-efficient 4, otherwise should be 3) so it measures hardware flops (used for HFU)
        """

        tokens_sum = 1  # noqa not used
        delta_time = 1  # noqa not used

        def _inner(config, model_size, seq_len):
            hardware_flops = True
            coef = 4 if hardware_flops else 3
            return (
                2 * coef * model_size * seq_len
                + 2 * 2 * coef * config.num_hidden_layers * config.hidden_size * seq_len**2
            ) / 1e12

        tflos = sum(_inner(self.config, self.model_size, seqlen) for seqlen in batch_seqlens)
        return tflos

    def _estimate_qwen2_flops(self, tokens_sum, batch_seqlens, delta_time):
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
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_deepseek_v3_flops(self, tokens_sum, batch_seqlens, delta_time):
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
        dense_N_flops = 6 * moe_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen * num_hidden_layers

        attn_qkv_flops = 12 * seqlen_square_sum * q_head_dim * num_query_heads
        # all_layer & all_token fwd & bwk flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12

        return flops_achieved

    def _estimate_qwen2_moe_flops(self, tokens_sum, batch_seqlens, delta_time):
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
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_gemma3_flops(self, tokens_sum, batch_seqlens, delta_time):
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
        # Gemma3 uses GeGLU (gelu_pytorch_tanh), having 3 matrices in MLP (inherited from Gemma2MLP)
        mlp_N = hidden_size * intermediate_size * 3
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer parm
        dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        # Gemma3 alternates between full and sliding window attention based on layer_types
        seqlen_square_sum = 0

        layer_types = getattr(self.config, "layer_types", None)
        sliding_window = getattr(self.config, "sliding_window", 1024)  # default 1024
        # default pattern: every 6th layer is full
        sliding_window_pattern = getattr(self.config, "sliding_window_pattern", 6)

        # If layer_types is not provided, generate it based on sliding_window_pattern
        if layer_types is None and sliding_window is not None and sliding_window_pattern is not None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(num_hidden_layers)
            ]

        if layer_types:
            # Calculate attention flops per layer based on attention type
            for layer_idx in range(num_hidden_layers):
                is_sliding = False
                if layer_types and layer_idx < len(layer_types):
                    is_sliding = layer_types[layer_idx] == "sliding_attention"

                for seqlen in batch_seqlens:
                    if is_sliding and sliding_window:
                        # Sliding window limits each token to attend to at most window_size tokens
                        effective_seqlen = min(seqlen, sliding_window)
                        seqlen_square_sum += seqlen * effective_seqlen
                    else:
                        # Full attention
                        seqlen_square_sum += seqlen * seqlen
        else:
            # If no layer_types config, assume all layers use full attention
            for seqlen in batch_seqlens:
                seqlen_square_sum += seqlen * seqlen
            seqlen_square_sum *= num_hidden_layers

        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def _estimate_apertus_flops(self, tokens_sum, batch_seqlens, delta_time):
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

        # Apertus MLP with XIELU activation uses only 2 linear layers (up_proj, down_proj)
        # No gate_proj for XIELU, unlike SwiGLU which has 3 layers
        mlp_N = hidden_size * intermediate_size * 2
        attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)

        # ApertusConfig has qk_norm defaulting to True.
        # This adds params for q_norm (on H) and k_norm (on num_kv_heads * head_dim)
        qk_norm_params_per_layer = hidden_size + num_key_value_heads * head_dim  # q_norm + k_norm

        emd_and_lm_head_N = vocab_size * hidden_size * 2
        # non-attn all_layer params
        dense_N = (mlp_N + attn_linear_N + qk_norm_params_per_layer) * num_hidden_layers + emd_and_lm_head_N
        # non-attn all_layer & all_token fwd & bwd flops
        dense_N_flops = 6 * dense_N * tokens_sum

        # attn all_layer & all_token fwd & bwd flops
        seqlen_square_sum = 0
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

        # all_layer & all_token fwd & bwd flops
        flops_all_token = dense_N_flops + attn_qkv_flops
        flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
        return flops_achieved

    def estimate_flops(self, batch_seqlens, delta_time):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the
                current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = self.estimate_func.get(self.config.model_type, self._estimate_dense_decoder_transformer_tflos)
        estimated_flops = func(tokens_sum, batch_seqlens, delta_time)
        return estimated_flops
