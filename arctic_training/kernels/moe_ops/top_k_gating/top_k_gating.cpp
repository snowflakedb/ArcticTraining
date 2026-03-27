// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "top_k_gating.h"
#include <c10/cuda/CUDAStream.h>

#define DISPATCH_TOP_K_GATING(T_TYPE, C_TYPE)                   \
    if (logits.options().dtype() == torch::T_TYPE) {            \
        launch_top_k_gating((int32_t*)expert_counts.data_ptr(), \
                            (float*)scores.data_ptr(),          \
                            (int32_t*)assignments.data_ptr(),   \
                            (int32_t*)offsets.data_ptr(),       \
                            (C_TYPE*)logits.data_ptr(),   \
                            (C_TYPE*)logits_out.data_ptr(),   \
                            n_tokens,                           \
                            n_experts,                          \
                            n_top_k,                            \
                            at::cuda::getCurrentCUDAStream());  \
        return;                                                 \
    }

/*
Perform softmax plus atomics in order to do first pass of top_k_gating.
*/
void top_k_gating(torch::Tensor& expert_counts,
                  torch::Tensor& scores,
                  torch::Tensor& assignments,
                  torch::Tensor& offsets,
                  torch::Tensor& logits,
                  torch::Tensor& logits_out
                //   torch::Tensor& batch_metadata
                  )
{
    const int32_t n_tokens = scores.size(0);
    const int32_t n_top_k = scores.size(1);

    // Should have the same buffer size for scores, offsets, and assignments
    TORCH_CHECK(n_tokens == offsets.size(0));
    TORCH_CHECK(n_tokens == logits.size(0));
    TORCH_CHECK(n_tokens == assignments.size(0));

    TORCH_CHECK(n_top_k == offsets.size(1));
    TORCH_CHECK(n_top_k == assignments.size(1));

    TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32);
    TORCH_CHECK(scores.scalar_type() == torch::kFloat);
    TORCH_CHECK(assignments.scalar_type() == torch::kInt32);
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32);

    const int32_t n_experts = logits.size(1);
    // const RaggedBatchDescriptor* batch_metadata_ptr =
    //     reinterpret_cast<const RaggedBatchDescriptor*>(batch_metadata.data_ptr());

    DISPATCH_TOP_K_GATING(kFloat, float)
    DISPATCH_TOP_K_GATING(kHalf, __half)
#ifdef BF16_AVAILABLE
    DISPATCH_TOP_K_GATING(kBFloat16, __nv_bfloat16)
#endif

    TORCH_CHECK(false, "Unsupported dtype for logits in top_k_gating");
}

#define DISPATCH_TOP_K_GATING_WITH_REPLAY(T_TYPE, C_TYPE)                   \
    if (logits.options().dtype() == torch::T_TYPE) {            \
        topk_gating_replay((int32_t*)expert_counts.data_ptr(), \
                            (float*)scores.data_ptr(),          \
                            (int32_t*)replay_assignments.data_ptr(),   \
                            (int32_t*)offsets.data_ptr(),       \
                            (C_TYPE*)logits.data_ptr(),   \
                            (C_TYPE*)logits_out.data_ptr(),   \
                            n_tokens,                           \
                            n_experts,                          \
                            n_top_k,                            \
                            at::cuda::getCurrentCUDAStream());  \
        return;                                                 \
    }

void top_k_gating_with_replay(torch::Tensor& expert_counts,
                  torch::Tensor& scores,
                  torch::Tensor& replay_assignments,
                  torch::Tensor& offsets,
                  torch::Tensor& logits,
                  torch::Tensor& logits_out)
{
    const int32_t n_tokens = scores.size(0);
    const int32_t n_experts = logits.size(1);
    const int32_t n_top_k = scores.size(1);

    // Should have the same buffer size for scores, offsets, and assignments
    TORCH_CHECK(n_tokens == offsets.size(0));
    TORCH_CHECK(n_tokens == logits.size(0));
    TORCH_CHECK(n_tokens == replay_assignments.size(0));

    TORCH_CHECK(n_top_k == offsets.size(1));
    TORCH_CHECK(n_top_k == replay_assignments.size(1));

    TORCH_CHECK(expert_counts.scalar_type() == torch::kInt32);
    TORCH_CHECK(scores.scalar_type() == torch::kFloat);
    TORCH_CHECK(replay_assignments.scalar_type() == torch::kInt32);
    TORCH_CHECK(offsets.scalar_type() == torch::kInt32);

    DISPATCH_TOP_K_GATING_WITH_REPLAY(kFloat, float)
    DISPATCH_TOP_K_GATING_WITH_REPLAY(kHalf, __half)
#ifdef BF16_AVAILABLE
    DISPATCH_TOP_K_GATING_WITH_REPLAY(kBFloat16, __nv_bfloat16)
#endif
    TORCH_CHECK(false, "Unsupported dtype for logits in top_k_gating_with_replay");
}


#define DISPATCH_TOPK_MOE_GATING_BWD(T_TYPE, C_TYPE)                   \
    if (logits.options().dtype() == torch::T_TYPE) {            \
        launch_topk_moe_gating_bwd((float*)scores_grad.data_ptr(),          \
                                   (C_TYPE*)logits_grad.data_ptr(),      \
                                   (C_TYPE*)logits.data_ptr(),   \
                                   (const int32_t*)assignments.data_ptr(),   \
                                   logits.size(1),                          \
                                   logits.size(0),                          \
                                   n_top_k,                            \
                                   at::cuda::getCurrentCUDAStream());  \
        return;                                                 \
    }

void top_k_gating_bwd(torch::Tensor& logits_grad,
                    torch::Tensor& scores_grad,
                    torch::Tensor& logits,
                    torch::Tensor& assignments)
{
    const int32_t n_tokens = scores_grad.size(0);
    const int32_t n_experts = logits.size(1);
    const int32_t n_top_k = scores_grad.size(1);

    TORCH_CHECK(n_tokens == logits_grad.size(0));
    TORCH_CHECK(n_tokens == logits.size(0));
    TORCH_CHECK(n_tokens == assignments.size(0));

    TORCH_CHECK(n_experts == logits_grad.size(1));
    TORCH_CHECK(n_experts == logits.size(1));

    TORCH_CHECK(assignments.scalar_type() == torch::kInt32);

    DISPATCH_TOPK_MOE_GATING_BWD(kFloat, float)
    DISPATCH_TOPK_MOE_GATING_BWD(kHalf, __half)
#ifdef BF16_AVAILABLE
    DISPATCH_TOPK_MOE_GATING_BWD(kBFloat16, __nv_bfloat16)
#endif
    TORCH_CHECK(false, "Unsupported dtype for logits in top_k_gating_bwd");
}
