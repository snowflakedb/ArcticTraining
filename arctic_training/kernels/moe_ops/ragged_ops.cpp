// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include <torch/extension.h>

#include "moe_gather.h"
#include "moe_scatter.h"
#include "top_k_gating.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // moe_gather.h
    m.def("moe_gather", &moe_gather, "MoE gather for top-1-gating.");
    m.def("moe_gather_backward", &moe_gather_backward, "Backward for MoE gather for top-1-gating.");

    // moe_scatter.h
    m.def("moe_scatter", &moe_scatter, "MoE scatter for top-1-gating.");
    m.def("moe_scatter_backward", &moe_scatter_backward, "Backward for MoE scatter for top-1-gating.");

    // top_k_gating.h
    m.def("top_k_gating", &top_k_gating, "Top-k gating for MoE with ragged batch awareness.");
    m.def("top_k_gating_with_replay", &top_k_gating_with_replay, "Top-k gating with replay for MoE with ragged batch awareness.");
    m.def("top_k_gating_bwd", &top_k_gating_bwd, "Backward for top-k gating for MoE with ragged batch awareness.");
}
