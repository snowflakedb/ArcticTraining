// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include "moe_gather.cuh"

/*
Re-gather the outputs of MoE and scale them by the gating score.
*/
void moe_gather(torch::Tensor& layer_output,
                const torch::Tensor& moe_output,
                const torch::Tensor& scores,
                const torch::Tensor& mapped_slots,
                const torch::Tensor& expert_counts,
                const bool normalize_scales);

void moe_gather_backward(torch::Tensor& grad_layer_output,
                         torch::Tensor& grad_scores,
                         torch::Tensor& grad_moe_output,
                         torch::Tensor& moe_output,
                         const torch::Tensor& scores,
                         const torch::Tensor& mapped_slots);
