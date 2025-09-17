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


import torch

import triton
import triton.autotune
import triton.Config
import triton.jit
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "NUM_SM": 84,
            }
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "NUM_SM": 128,
            }
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "NUM_SM": 84,
            }
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "NUM_SM": 128,
            }
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "NUM_SM": num_sms(),
            }
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "NUM_SM": num_sms(),
            }
        ),
    ],
    key=["group_size"],
)
@triton.jit
def grouped_matmul_kernel(
    # device tensor of matrices pointers
    a_ptr,
    b_ptr,
    c_ptr,
    rows_cumsum_ptr,
    m,
    k,
    n,
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    lda = k
    ldb = n
    ldc = n
    total_m = tl.load(rows_cumsum_ptr + group_size - 1)
    cur_m = tl.load(rows_cumsum_ptr)
    num_m_tiles = tl.cdiv(total_m, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(n, BLOCK_SIZE_N)
    num_tiles = num_m_tiles * num_n_tiles

    while tile_idx < num_tiles:
        # figure out tile coordinates
        tile_m_idx = tile_idx // num_n_tiles
        tile_n_idx = tile_idx % num_n_tiles

        g = (tile_m_idx * BLOCK_SIZE_M) // cur_m
        gb_ptr = b_ptr + g * k * n
        cur_m = tl.load(rows_cumsum_ptr + g)

        # do regular gemm here
        offs_acm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bcn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + offs_acm[:, None] * lda + offs_k[None, :]
        b_ptrs = gb_ptr + offs_k[:, None] * ldb + offs_bcn[None, :]
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
            # hint to Triton compiler to do proper loop pipelining
            tl.multiple_of(a_ptrs, [16, 16])
            tl.multiple_of(b_ptrs, [16, 16])
            # assume full tile for now
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K * ldb
        c = accumulator.to(tl.float16)
        c_ptr_g = c_ptr + ldc * offs_acm[:, None] + offs_bcn[None, :]
        # assumes full tile for now
        tl.store(c_ptr_g, c)
        # go to the next tile by advancing NUM_SM
        tile_idx += NUM_SM


def group_gemm_fn(A, B, rows_cumsum):
    group_size = rows_cumsum.size(0)
    C = torch.zeros((rows_cumsum[-1], B.size(-1)), device=A.device, dtype=A.dtype)
    grid = lambda META: (META["NUM_SM"],)  # noqa
    grouped_matmul_kernel[grid](
        A,
        B,
        C,
        rows_cumsum,
        A.shape[0],
        A.shape[1],
        B.shape[-1],
        group_size,
    )

    return C


def torch_group_gemm_fn(A, B, rows_cumsum):
    C = torch.zeros((rows_cumsum[-1], B.size(-1)), device=A.device, dtype=A.dtype)
    for i in range(len(rows_cumsum)):
        start = 0 if i == 0 else rows_cumsum[i - 1]
        end = rows_cumsum[i]
        C[start:end, :] = torch.matmul(A[start:end, :], B[i])
    return C
