// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "moe_gather.h"
#include <c10/cuda/CUDAStream.h>

#define DISPATCH_MOE_GATHER(T_TYPE, C_TYPE)                        \
    if (layer_output.options().dtype() == torch::T_TYPE) {         \
        launch_moe_gather((C_TYPE*)layer_output.data_ptr(),        \
                          (const C_TYPE*)moe_output.data_ptr(),    \
                          (const float*)scores.data_ptr(),         \
                          (const int32_t*)mapped_slots.data_ptr(), \
                          (int32_t*)expert_count.data_ptr(),       \
                          n_channels,                              \
                          n_experts,                               \
                          n_tokens,                                \
                          n_top_k,                                 \
                          normalize_scales,                        \
                          at::cuda::getCurrentCUDAStream());       \
        return;                                                    \
    }

/*
Re-gather the outputs of MoE and scale them by the gating score.
*/
void moe_gather(torch::Tensor& layer_output,
                const torch::Tensor& moe_output,
                const torch::Tensor& scores,
                const torch::Tensor& mapped_slots,
                const torch::Tensor& expert_count,
                const bool normalize_scales)
{
    const int32_t n_channels = layer_output.size(1);
    const int32_t n_experts = expert_count.size(0);
    const int32_t n_tokens = layer_output.size(0);
    const int32_t n_top_k = mapped_slots.size(1);

    // TORCH_CHECK(moe_output.size(0) == n_tokens * n_top_k);
    TORCH_CHECK(moe_output.size(1) == n_channels);
    TORCH_CHECK(scores.size(0) == n_tokens);
    TORCH_CHECK(mapped_slots.size(0) == n_tokens);

    TORCH_CHECK(scores.size(1) == n_top_k);

    TORCH_CHECK(layer_output.scalar_type() == moe_output.scalar_type());
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32);
    TORCH_CHECK(mapped_slots.scalar_type() == torch::kInt32);
    TORCH_CHECK(expert_count.scalar_type() == torch::kInt32);

    DISPATCH_MOE_GATHER(kHalf, __half);

#ifdef BF16_AVAILABLE
    DISPATCH_MOE_GATHER(kBFloat16, __nv_bfloat16);
#endif

    TORCH_CHECK(false, "Unsupported data type for MoE gather");
}
#define DISPATCH_MOE_GATHER_BWD(T_TYPE, C_TYPE)                        \
    if (grad_layer_output.options().dtype() == torch::T_TYPE) {         \
        launch_topk_moe_gather_bwd((C_TYPE*)grad_layer_output.data_ptr(),        \
                          (float*)grad_scores.data_ptr(),         \
                          (C_TYPE*)grad_moe_output.data_ptr(),    \
                          (C_TYPE*)moe_output.data_ptr(),    \
                          (const float*)scores.data_ptr(),         \
                          (const int32_t*)mapped_slots.data_ptr(), \
                          n_channels,                              \
                          n_tokens,                                \
                          n_top_k,                                 \
                          at::cuda::getCurrentCUDAStream());       \
        return;                                                    \
    }

void moe_gather_backward(torch::Tensor& grad_layer_output,
                         torch::Tensor& grad_scores,
                         torch::Tensor& grad_moe_output,
                         torch::Tensor& moe_output,
                         const torch::Tensor& scores,
                         const torch::Tensor& mapped_slots)
{
    const int32_t n_channels = grad_layer_output.size(1);
    const int32_t n_tokens = grad_layer_output.size(0);
    const int32_t n_top_k = mapped_slots.size(1);

    // TORCH_CHECK(grad_moe_output.size(0) == n_tokens * n_top_k);
    TORCH_CHECK(grad_moe_output.size(1) == n_channels);
    TORCH_CHECK(scores.size(0) == n_tokens);
    TORCH_CHECK(mapped_slots.size(0) == n_tokens);

    TORCH_CHECK(scores.size(1) == n_top_k);

    TORCH_CHECK(grad_layer_output.scalar_type() == grad_moe_output.scalar_type());
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32);
    TORCH_CHECK(mapped_slots.scalar_type() == torch::kInt32);

    DISPATCH_MOE_GATHER_BWD(kHalf, __half);
    DISPATCH_MOE_GATHER_BWD(kFloat, float);
    #ifdef BF16_AVAILABLE
    DISPATCH_MOE_GATHER_BWD(kBFloat16, __nv_bfloat16);
    #endif
}
