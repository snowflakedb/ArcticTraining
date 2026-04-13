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


import os

from .builder import CUDAOpBuilder
from .builder import installed_cuda_version


class RaggedOpsBuilder(CUDAOpBuilder):
    BUILD_VAR = "AT_BUILD_RAGGED_OPS"
    NAME = "ragged_ops"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"arctic_training.{self.NAME}"

    def is_compatible(self, verbose=False):
        try:
            import torch
        except ImportError:
            if verbose:
                self.warning("Please install torch if trying to pre-compile arctic_training kernels")
            return False

        cuda_okay = True
        if torch.cuda.is_available():  # ignore-cuda
            sys_cuda_major, _ = installed_cuda_version()
            torch_cuda_major = int(torch.version.cuda.split(".")[0])
            cuda_capability = torch.cuda.get_device_properties(0).major  # ignore-cuda
            if cuda_capability < 6:
                if verbose:
                    self.warning("NVIDIA Inference is only supported on Pascal and newer architectures")
                cuda_okay = False
            if cuda_capability >= 8:
                if torch_cuda_major < 11 or sys_cuda_major < 11:
                    if verbose:
                        self.warning("On Ampere and higher architectures please use CUDA 11+")
                    cuda_okay = False
        return super().is_compatible(verbose) and cuda_okay

    def filter_ccs(self, ccs):
        ccs_retained = []
        ccs_pruned = []
        for cc in [cc.split(".") for cc in ccs]:
            if int(cc[0]) >= 8:
                # Blocked flash has a dependency on Ampere + newer
                ccs_retained.append(cc)
            else:
                ccs_pruned.append(cc)
        if len(ccs_pruned) > 0:
            self.warning(f"Filtered compute capabilities {ccs_pruned}")
        return ccs_retained

    def get_prefix(self):
        ai_path = self._src_path("arctic_training")
        return "arctic_training" if os.path.isdir(ai_path) else ".."

    def sources(self):
        sources = [
            "arctic_training/kernels/moe_ops/ragged_ops.cpp",
            "arctic_training/kernels/moe_ops/moe_scatter/moe_scatter.cpp",
            "arctic_training/kernels/moe_ops/moe_scatter/moe_scatter_cuda.cu",
            "arctic_training/kernels/moe_ops/moe_gather/moe_gather.cpp",
            "arctic_training/kernels/moe_ops/moe_gather/moe_gather_cuda.cu",
            "arctic_training/kernels/moe_ops/top_k_gating/top_k_gating.cpp",
            "arctic_training/kernels/moe_ops/top_k_gating/top_k_gating_cuda.cu",
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources

    def extra_ldflags(self):
        return []

    def include_paths(self):
        sources = [
            "arctic_training/kernels/includes",
            "arctic_training/kernels/moe_ops/moe_gather",
            "arctic_training/kernels/moe_ops/moe_scatter",
            "arctic_training/kernels/moe_ops/top_k_gating",
        ]

        prefix = self.get_prefix()
        sources = [os.path.join(prefix, src) for src in sources]
        return sources
