// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "top_k_gating.cuh"

/*
Perform softmax plus atomics to get token mapping.
*/
void top_k_gating(torch::Tensor& expert_counts,
                  torch::Tensor& scores,
                  torch::Tensor& assignments,
                  torch::Tensor& offsets,
                  torch::Tensor& logits,
                  torch::Tensor& logits_out
                //   torch::Tensor& batch_metadata
                );

void top_k_gating_bwd(torch::Tensor& logits_grad,
                    torch::Tensor& scores_grad,
                    torch::Tensor& logits,
                    torch::Tensor& assignments
                    );

void top_k_gating_with_replay(torch::Tensor& expert_counts,
                  torch::Tensor& scores,
                  torch::Tensor& replay_assignments,
                  torch::Tensor& offsets,
                  torch::Tensor& logits,
                  torch::Tensor& logits_out);
