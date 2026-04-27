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


def AlltoAllV(*args, **kwargs):
    return AlltoAllV_Func.apply(*args, **kwargs)


class AlltoAllFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, x):
        x = x.contiguous()
        # y = torch.empty_like(x)
        ctx.group = group
        # dist.all_to_all_single(y, x, group=group)
        return x
        # return y

    @staticmethod
    def backward(ctx, grad_output):
        return (None, AlltoAllFunction.apply(ctx.group, grad_output))


def AlltoAll(*args, **kwargs):
    return AlltoAllFunction.apply(*args, **kwargs)


class CustomAlltoAllFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, comm, x, counts, max_count):
        ctx.comm = comm

        receive_counts = torch.empty_like(counts)
        torch.distributed.all_to_all_single(receive_counts, counts)

        y = comm.all_to_all(x, counts=counts, receive_counts=receive_counts, max_count=max_count)[0]

        ctx.save_for_backward(receive_counts, max_count, counts)
        return y, receive_counts

    @staticmethod
    def backward(ctx, *grad_outputs):
        receive_counts, max_count, counts = ctx.saved_tensors
        grad_output = grad_outputs[0].contiguous()
        grad_input = ctx.comm.all_to_all(
            grad_output, counts=receive_counts, receive_counts=counts, max_count=max_count
        )[0]
        return (None, grad_input, None, None)


def CustomAlltoAll(*args, **kwargs):
    return CustomAlltoAllFunction.apply(*args, **kwargs)
