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

# This file was original taken from the following:
# https://github.com/MoonshotAI/MoBA/tree/61e456bc956c5a25fd9c84e5496b661329cb1b72
# Modification may have been made by Snowflake

from dataclasses import dataclass


@dataclass
class MoBAConfig:
    moba_chunk_size: int
    moba_topk: int
