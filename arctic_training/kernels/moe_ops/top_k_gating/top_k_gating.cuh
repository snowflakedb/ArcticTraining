// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include "ds_kernel_utils.h"
#include "ragged_dtypes.h"

namespace gating {
constexpr int unassigned = -1;
}  // namespace gating

template <typename T>
void launch_top_k_gating(int32_t* expert_counts,
                         float* scores,
                         int32_t* assignments,
                         int32_t* offsets,
                         T* logits,
                         T* logits_out,
                        //  const RaggedBatchDescriptor* batch_metadata,
                         const int32_t n_tokens,
                         const int32_t n_experts,
                         const int32_t n_top_k,
                         cudaStream_t stream);

template <typename T>
void launch_topk_moe_gating_bwd(
                        float* scores_grad,
                        T* logits_grad,
                        T* logits,
                        const int32_t* assignments,
                        const int32_t n_experts,
                        const int32_t n_tokens,
                        const int32_t n_top_k,
                        cudaStream_t stream);

template <typename T>
void topk_gating_replay(int32_t* expert_counts,
                         float* scores,
                         int32_t* replay_assignments,
                         int32_t* offsets,
                         T* logits,
                         T* logits_out,
                         const int32_t n_tokens,
                         const int32_t n_experts,
                         const int32_t n_top_k,
                         cudaStream_t stream);
