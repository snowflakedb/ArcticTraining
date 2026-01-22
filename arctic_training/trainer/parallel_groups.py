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

from torch import distributed as dist


class ParallelGroups:
    def __init__(
        self,
        data_parallel_size=1,
        sequence_parallel_size=1,
        expert_parallel_size=1,
        use_model_parallel_for_experts=False,
    ):
        self.data_parallel_size = data_parallel_size
        self.sequence_parallel_size = sequence_parallel_size
        self.expert_parallel_size = expert_parallel_size
        self.create_groups(use_model_parallel_for_experts)

    def create_groups(self, use_model_parallel_for_experts):

        world_size = dist.get_world_size()
        if self.data_parallel_size * self.sequence_parallel_size * self.expert_parallel_size != world_size:
            raise ValueError(
                "{self.data_parallel_size * self.sequence_parallel_size * self.expert_parallel_size=} != {world_size}"
            )

        # print(f"creating device_mesh {self.data_parallel_size=}/{self.sequence_parallel_size=}/{self.expert_parallel_size=}")
        device_mesh = dist.init_device_mesh(
            "cuda",
            (
                self.data_parallel_size,
                self.sequence_parallel_size,
                self.expert_parallel_size,
            ),
            mesh_dim_names=(
                "data_parallel",
                "sequence_parallel",
                "expert_parallel",
            ),
        )
        self.data_parallel_group = device_mesh.get_group(mesh_dim="data_parallel")
        self.sequence_parallel_group = device_mesh.get_group(mesh_dim="sequence_parallel")
        self.expert_parallel_group = device_mesh.get_group(mesh_dim="expert_parallel")

        # ep_dp_mp_device_mesh = dist.init_device_mesh(
        #     "cuda",
        #     (
        #         self.data_parallel_size
        #         // self.expert_parallel_size
        #         // (self.model_parallel_size if use_model_parallel_for_experts else 1),
        #         self.expert_parallel_size,
        #         self.model_parallel_size if use_model_parallel_for_experts else 1,
        #     ),
        #     mesh_dim_names=("expert_data_parallel", "expert_parallel", "expert_model_parallel"),
        # )
        # self.expert_data_parallel_group = ep_dp_mp_device_mesh.get_group(mesh_dim="expert_data_parallel")
        # self.expert_model_parallel_group = ep_dp_mp_device_mesh.get_group(mesh_dim="expert_model_parallel")
        # if use_model_parallel_for_experts:
        #     assert dist.get_process_group_ranks(group=self.model_parallel_group) == dist.get_process_group_ranks(
        #         group=self.expert_model_parallel_group
        #     ), (
        #         "Model parallel group and expert model parallel group must be on the same ranks if model parallelism"
        #         " is used for experts."
        #     )

    @property
    def get_data_parallel_rank(self):
        return dist.get_rank(group=self.data_parallel_group)

    # @property
    # def get_model_parallel_rank(self):
    #     return dist.get_rank(group=self.model_parallel_group)

    @property
    def get_sequence_parallel_rank(self):
        return dist.get_rank(group=self.sequence_parallel_group)

    @property
    def get_expert_parallel_rank(self):
        return dist.get_rank(group=self.expert_parallel_group)

    @property
    def get_data_parallel_world_size(self):
        return dist.get_world_size(group=self.data_parallel_group)

    # @property
    # def get_model_parallel_world_size(self):
    #     return dist.get_world_size(group=self.model_parallel_group)

    @property
    def get_sequence_parallel_world_size(self):
        return dist.get_world_size(group=self.sequence_parallel_group)

    @property
    def get_expert_parallel_world_size(self):
        return dist.get_world_size(group=self.expert_parallel_group)

    @property
    def get_data_parallel_group(self):
        return self.data_parallel_group

    # @property
    # def get_model_parallel_group(self):
    #     return self.model_parallel_group

    @property
    def get_sequence_parallel_group(self):
        return self.sequence_parallel_group

    @property
    def get_expert_parallel_group(self):
        return self.expert_parallel_group

    # @property
    # def get_expert_data_parallel_group(self):
    #     return self.expert_data_parallel_group
