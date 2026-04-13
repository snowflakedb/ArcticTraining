// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "conversion_utils.h"
#include "memory_access_utils.h"
#include "reduction_utils.h"
#include "top_k_gating.cuh"
#include "top_k_utils.h"

using ROp = reduce::ROpType;

template <typename T, int TOP_K>
__global__ void top_k_replay_kernel(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* replay_assignments,
                                    int32_t* offsets,
                                    T* logits,
                                    T* logits_out,
                                    // const RaggedBatchDescriptor* batch_metadata,
                                    const int32_t n_tokens,
                                    const int32_t n_experts)
{
    const int32_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t expert_idx = threadIdx.x;
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    if (token_idx >= n_tokens) {
        return;
    }

    T* token_logits = logits + token_idx * n_experts;
    T* token_logits_out = logits_out + token_idx * n_experts;

    float logit_val;
    if (expert_idx < n_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }
    float reduce_val = logit_val;

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.

   reduce::block<ROp::Max>(tb, warp, reduce_val);

    const float max_logit = reduce_val;
    float softlogit = __expf(logit_val - max_logit);
    float softmax_sum = softlogit;
    reduce::block<ROp::Add>(tb, warp, softmax_sum);
    if (expert_idx < n_experts)
        token_logits_out[expert_idx] = conversion::to<T>(softlogit / softmax_sum);

    for (int i = 0; i < TOP_K; ++i) {
        int32_t local_assigned_experts;
        float local_assigned_logits;
        local_assigned_experts = replay_assignments[token_idx * TOP_K + i];
        local_assigned_logits = token_logits[local_assigned_experts];
        const float softmax = __expf(local_assigned_logits - max_logit) / softmax_sum;

        scores[token_idx * TOP_K + i] = softmax;
        offsets[token_idx * TOP_K + i] = atomicAdd(expert_counts + local_assigned_experts, 1);
    }
}

template <typename T, int TOP_K>
__global__ void top_k_gating_kernel(int32_t* expert_counts,
                                    float* scores,
                                    int32_t* assignments,
                                    int32_t* offsets,
                                    T* logits,
                                    T* logits_out,
                                    // const RaggedBatchDescriptor* batch_metadata,
                                    const int32_t n_tokens,
                                    const int32_t n_experts)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;
    const int32_t max_warps = 1024 / hw_warp_size;

    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    // if (token_idx >= batch_metadata->n_tokens) {
    if (token_idx >= n_tokens) {
        return;
    }

    T* token_logits = logits + token_idx * n_experts;
    T* token_logits_out = logits_out + token_idx * n_experts;

    float logit_val;
    if (expert_idx < n_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
    } else {
        reduce::init<ROp::Max>(&logit_val);
    }
    float reduce_val = logit_val;

    int32_t local_assigned_experts[TOP_K];
    float local_assigned_logits[TOP_K];

    // Training code tends to use ``torch.argmax`` to select the expert, which
    // which has ties broken by the lower index. Since our fused comparison algorithm
    // breaks ties by the higher index (since it's the lower 32-bits of the 64-bit
    // comparison), we invert the expert index to break ties by the lower index.
    int32_t inverted_expert = n_experts - expert_idx - 1;

    // Find the top k logits
    for (int i = 0; i < TOP_K; ++i) {
        const reduce::IdxReduceResult res =
            reduce::idx_reduce<ROp::Max, max_warps>(tb, warp, reduce_val, inverted_expert);
        local_assigned_experts[i] = n_experts - res.idx - 1;
        local_assigned_logits[i] = res.val;

        // Set the max logit to -inf so that it is not selected again
        if (threadIdx.x == n_experts - res.idx - 1) { reduce::init<ROp::Max>(&reduce_val); }
    }

    const float max_logit = local_assigned_logits[0];
    float softlogit = __expf(logit_val - max_logit);
    float softmax_sum = softlogit;
    reduce::block<ROp::Add>(tb, warp, softmax_sum);
    if (expert_idx < n_experts)
        token_logits_out[expert_idx] = conversion::to<T>(softlogit / softmax_sum);

    for (int i = 0; i < TOP_K; ++i) {
        const float softmax = __expf(local_assigned_logits[i] - max_logit) / softmax_sum;

        if (threadIdx.x == 0) {
            scores[token_idx * TOP_K + i] = softmax;
            assignments[token_idx * TOP_K + i] = local_assigned_experts[i];
            offsets[token_idx * TOP_K + i] =
                atomicAdd(expert_counts + local_assigned_experts[i], 1);
        }
    }
}

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
                         cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    const dim3 block(((n_experts + hw_warp_size - 1) / hw_warp_size) * hw_warp_size);

    TOP_K_SWITCH(n_top_k, [&] {
        // top_k_gating_kernel<T, CONST_TOP_K><<<grid, block, 0, stream>>>(
        //     expert_counts, scores, assignments, offsets, logits, batch_metadata, n_experts);
        top_k_gating_kernel<T, CONST_TOP_K><<<grid, block, 0, stream>>>(
            expert_counts, scores, assignments, offsets, logits, logits_out, n_tokens, n_experts);
    });
}

// #define INSTANTIATE_top_k_KERNEL(T)                                                   \
//     template void launch_top_k_gating<T>(int32_t * expert_counts,                     \
//                                          float* scores,                               \
//                                          int32_t* assignments,                        \
//                                          int32_t* offsets,                            \
//                                          const T* logits,                             \
//                                          const RaggedBatchDescriptor* batch_metadata, \
//                                          const int32_t n_tokens,                      \
//                                          const int32_t n_experts,                     \
//                                          const int32_t n_top_k,                       \
//                                          cudaStream_t stream);
#define INSTANTIATE_top_k_KERNEL(T)                                                   \
    template void launch_top_k_gating<T>(int32_t* expert_counts,                     \
                                         float* scores,                               \
                                         int32_t* assignments,                        \
                                         int32_t* offsets,                            \
                                         T* logits,                             \
                                         T* logits_out,                             \
                                         const int32_t n_tokens,                      \
                                         const int32_t n_experts,                     \
                                         const int32_t n_top_k,                       \
                                         cudaStream_t stream);
INSTANTIATE_top_k_KERNEL(float)
INSTANTIATE_top_k_KERNEL(__half)
#ifdef BF16_AVAILABLE
    INSTANTIATE_top_k_KERNEL(__nv_bfloat16)
#endif

template <typename T>
void topk_gating_replay(int32_t* expert_counts,
                        float* scores,
                        int32_t* replay_assignments,
                        int32_t* offsets,
                        T* logits,
                        T* logits_out,
                        // const RaggedBatchDescriptor* batch_metadata,
                        const int32_t n_tokens,
                        const int32_t n_experts,
                        const int32_t n_top_k,
                        cudaStream_t stream)
{
    constexpr int threads = 256;
    int blocks = (n_tokens + threads - 1) / threads;
    const dim3 grid(blocks);
    const dim3 block(threads);

    TOP_K_SWITCH(n_top_k, [&] {
        top_k_replay_kernel<T, CONST_TOP_K><<<grid, block, 0, stream>>>(
            expert_counts, scores, replay_assignments, offsets, logits, logits_out, n_tokens, n_experts);
    });
}

#define INSTANTIATE_top_k_REPLAY_KERNEL(T)                                                   \
    template void topk_gating_replay<T>(int32_t* expert_counts,                     \
                                         float* scores,                               \
                                         int32_t* replay_assignments,                        \
                                         int32_t* offsets,                            \
                                         T* logits,                             \
                                         T* logits_out,                             \
                                         const int32_t n_tokens,                      \
                                         const int32_t n_experts,                     \
                                         const int32_t n_top_k,                       \
                                         cudaStream_t stream);
INSTANTIATE_top_k_REPLAY_KERNEL(float)
INSTANTIATE_top_k_REPLAY_KERNEL(__half)
#ifdef BF16_AVAILABLE
    INSTANTIATE_top_k_REPLAY_KERNEL(__nv_bfloat16)
#endif


template<typename T, int TOP_K>
__global__ void top_k_gate_logits_bwd_kernel(T* logits_grad,
                                        float* scores_grad,
                                        const int32_t* assignment,
                                        T* logits,
                                        const int32_t n_experts,
                                        const int32_t n_tokens)
{
    const int32_t token_idx = blockIdx.x;
    const int32_t expert_idx = threadIdx.x;

    int32_t assigned_expert[TOP_K];

#pragma unroll
    for (int i = 0; i < TOP_K; i++)
        assigned_expert[i] = assignment[token_idx * TOP_K + i];
    // CG helpers
    cg::thread_block tb = cg::this_thread_block();
    cg::thread_block_tile<hw_warp_size> warp = cg::tiled_partition<hw_warp_size>(tb);

    // Padding tokens do not require
    if (token_idx >= n_tokens) {
        return;
    }

    T* token_logits = logits + token_idx * n_experts;
    T* token_logits_grad = logits_grad + token_idx * n_experts;

    float logit_val;
    float logit_grad_val;
    if (expert_idx < n_experts) {
        logit_val = conversion::to<float>(token_logits[expert_idx]);
        logit_grad_val = conversion::to<float>(token_logits_grad[expert_idx]);
    } else {
        reduce::init<ROp::Add>(&logit_val);
        reduce::init<ROp::Add>(&logit_grad_val);
    }


#pragma unroll
    for (int i = 0; i < TOP_K; i++)
    {
        if (assigned_expert[i] == expert_idx) {
            logit_grad_val += scores_grad[token_idx * TOP_K + i];
        }
    }
    float softmax_grad_sum = logit_val * logit_grad_val;
    reduce::block<ROp::Add>(tb, warp, softmax_grad_sum);
    logit_grad_val = logit_val * (logit_grad_val - softmax_grad_sum);
    if (expert_idx < n_experts)
        token_logits_grad[expert_idx] = conversion::to<T>(logit_grad_val);
}

template <typename T>
void launch_topk_moe_gating_bwd(
                        float* scores_grad,
                        T* logits_grad,
                        T* logits,
                        const int32_t* assignments,
                        const int32_t n_experts,
                        const int32_t n_tokens,
                        const int32_t n_top_k,
                        cudaStream_t stream)
{
    const dim3 grid(n_tokens);
    const dim3 block(((n_experts - 1) / hw_warp_size + 1) * hw_warp_size);

    TOP_K_SWITCH(n_top_k, [&] {
        top_k_gate_logits_bwd_kernel<T, CONST_TOP_K><<<grid, block, 0, stream>>>
            (logits_grad, scores_grad, assignments, logits, n_experts, n_tokens);
    });
}

#define INSTANTIATE_TOPK_MOE_GATING_BWD_FOR_TYPE(TYPE)                              \
    template void launch_topk_moe_gating_bwd<TYPE>(                     \
                                          float* scores_grad,           \
                                          TYPE* logits_grad,      \
                                          TYPE* logits,      \
                                          const int32_t* assignments,      \
                                          const int32_t n_experts,     \
                                          const int32_t n_tokens,       \
                                          const int32_t n_top_k,        \
                                          cudaStream_t stream);         \

INSTANTIATE_TOPK_MOE_GATING_BWD_FOR_TYPE(float)
INSTANTIATE_TOPK_MOE_GATING_BWD_FOR_TYPE(__half)

#ifdef BF16_AVAILABLE
INSTANTIATE_TOPK_MOE_GATING_BWD_FOR_TYPE(__nv_bfloat16)
#endif
