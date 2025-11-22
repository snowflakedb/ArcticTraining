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
import torch.distributed as dist


class AlltoAllV_Func(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        group,
        x,
        send_counts,
        recv_counts,
    ):
        ctx.group = group
        x_splits = send_counts.tolist()
        y_splits = recv_counts.tolist()
        x = x.contiguous()
        y = torch.empty(sum(y_splits), x.shape[1], dtype=x.dtype, device=x.device)
        dist.all_to_all_single(y, x, output_split_sizes=y_splits, input_split_sizes=x_splits, group=group)
        ctx.save_for_backward(send_counts, recv_counts)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        send_counts, recv_counts = ctx.saved_tensors
        return (None, AlltoAllV_Func.apply(ctx.group, grad_output, recv_counts, send_counts), None, None)


AlltoAllV = AlltoAllV_Func.apply
