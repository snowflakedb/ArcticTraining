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

from arctic_training.op_builder import CommBuilder

from .layout import Layout

ds_comm = None


class Comm:
    current_comm = None

    def __init__(self, layout: Layout, local_rank: int):
        global ds_comm
        if ds_comm is None:
            ds_comm = CommBuilder().load()

        self.ds_comm = ds_comm
        self._layout = layout
        self.my_rank = local_rank
        self.global_rank = torch.distributed.get_rank()
        self.group_size = layout._group_size

        self.global_ranks = layout.sibling_ranks(self.global_rank)
        self.local_ranks = list(range(self.group_size))
        self.rank_map = dict(zip(self.global_ranks, self.local_ranks))
        self.counts_pinned_data = torch.empty(1024, dtype=torch.int32, device="cpu", pin_memory=True)
        self.recv_counts_pinned_data = torch.empty(1024, dtype=torch.int32, device="cpu", pin_memory=True)

        print("Initializing comm ...")

    def all_reduce(self, val, inplace=True, async_op=False):
        val_sum = val if inplace else torch.empty_like(val)
        op = communicate_op(val, val_sum, async_op, op_type="all_reduce")
        return val_sum, op

    def all_gather(self, val, inplace=True, async_op=False):
        val_gather = torch.empty((self.group_size * val.size(0), *val.shape[1:]), device=val.device, dtype=val.dtype)
        op = communicate_op(val, val_gather, async_op, op_type="all_gather")
        return val_gather, op

    def all_to_all(self, val, counts=None, result=None, inplace=True, async_op=False):
        if counts is not None:
            receive_counts = torch.empty_like(counts)
            self.counts_pinned_data[: receive_counts.numel()].copy_(counts)
            torch.distributed.all_to_all_single(receive_counts, counts)
            max_count = counts.max()
            torch.distributed.all_reduce(max_count, op=torch.distributed.ReduceOp.MAX)

            max_count = max_count.item()
            receive_counts = self.recv_counts_pinned_data[: receive_counts.numel()].copy_(receive_counts)
            counts = self.counts_pinned_data[: counts.numel()]
        else:
            max_count = val.size(0) // self.group_size
            counts = torch.full((self.group_size,), max_count, device="cpu", dtype=torch.int32)
            receive_counts = counts

        result = result if result is not None else torch.empty_like(val)
        op = communicate_op(
            val,
            result,
            async_op,
            world_size=self.group_size,
            op_type="all_to_all",
            send_counts=counts,
            recv_counts=receive_counts,
            max_count=max_count,
        )
        return result, op

    def broadcast(self, val, inplace=True, async_op=False):
        val_bcst = torch.empty_like(val)
        op = communicate_op(val, val_bcst, async_op, op_type="broadcast")
        return val_bcst, op

    def barrier(self):
        ds_comm.wait_comm()
        ds_comm.barrier()

    @classmethod
    def get_current_comm(cls):
        if cls.current_comm is None:
            from arctic_training.kernels.comm.nccl import NcclComm

            cls.current_comm = NcclComm()
        return cls.current_comm


class communicate_op:
    def __init__(
        self,
        val,
        result,
        async_op,
        world_size=None,
        op_type="all_reduce",
        send_counts=None,
        recv_counts=None,
        max_count=None,
    ):
        if op_type == "all_reduce":
            ds_comm.allReduce(val, result, val.numel(), async_op)
        elif op_type == "all_gather":
            ds_comm.allGather(val, result, val.numel(), async_op)
        elif op_type == "all_to_all":
            ds_comm.alltoall(val, result, send_counts, recv_counts, max_count, async_op)
        elif op_type == "broadcast":
            ds_comm.broadcast(val, result, val.numel(), async_op)

    def wait(self):
        ds_comm.wait_comm()


def get_default_comm():
    return Comm.get_current_comm()
