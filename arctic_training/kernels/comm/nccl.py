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

import torch

from .comm import Comm
from .layout import Layout


class NcclComm(Comm):

    def __init__(self, layout: Layout, set_device: int = True):

        super().__init__(layout, int(os.getenv("LOCAL_RANK", "0")))

        dist_world_size = torch.distributed.get_world_size()

        self.comm_group = None
        self.parent_rank = 0
        if self.group_size < dist_world_size:
            self.parent_rank = self._layout.parent_rank(self.global_rank)
            for sranks in self._layout._sibling_ranks:
                comm_group = torch.distributed.new_group(sranks)
                if self.global_rank in sranks:
                    self.comm_group = comm_group

        if set_device:
            torch.cuda.set_device(f"cuda:{self.my_rank}")

        nccl_uid = torch.tensor([torch.cuda.nccl.unique_id()], dtype=torch.uint8, device=torch.cuda.current_device())
        torch.distributed.broadcast(nccl_uid, self.parent_rank, group=self.comm_group)
        self.ds_comm.init_nccl_comm(self.local_ranks, self.rank_map[self.global_rank], nccl_uid.to("cpu").squeeze(0))


class MPIComm(Comm):

    def __init__(self, layout=None, set_device=True):
        from mpi4py import MPI

        local_rank = int(MPI.COMM_WORLD.Get_rank())
        super().__init__(layout, local_rank)
        if set_device:
            torch.cuda.set_device(f"cuda:{self.my_rank}")
        self.ds_comm.init_comm_group(self.global_ranks, self.rank_map[self.global_rank])


def create_comm(layout=None, set_device=True, backend="nccl"):
    return (
        NcclComm(layout=layout, set_device=set_device)
        if backend == "nccl"
        else MPIComm(layout=layout, set_device=set_device)
    )
