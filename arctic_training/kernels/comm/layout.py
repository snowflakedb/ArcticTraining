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


class Layout:
    def __init__(self, g_size=None, stride=1, world_size=1):

        num_groups = world_size // g_size
        self._group_size = g_size
        self._sibling_ranks = []
        for gid in range(num_groups):
            if stride == 1:
                self._sibling_ranks.append([gid * g_size + i for i in range(g_size)])
            else:
                self._sibling_ranks.append([gid % stride + i * stride for i in range(g_size)])

    def sibling_ranks(self, rank):
        for sranks in self._sibling_ranks:
            if rank in sranks:
                break
        return sranks

    def parent_rank(self, rank):
        return self.sibling_ranks(rank)[0]
