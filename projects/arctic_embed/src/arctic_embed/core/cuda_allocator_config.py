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

import logging
import os
from typing import Dict
from typing import NamedTuple
from typing import Optional

logger = logging.getLogger(__name__)


class CudaAllocatorConfig(NamedTuple):
    """Configuration for the default `native` PyTorch GPU memory allocator.

    See: https://pytorch.org/docs/stable/notes/cuda.html#memory-management

    NOTE: When batch size or sequence length varies during a training job,
    both `roundup_power2_divisions` and `expandable_segments` have been tested
    to fix OOM issues without performance degredation in at least one case.

    NOTE: Currently only a single value roundup_power2_divisions is implemented
    in this wrapper, although the setting supports more customization than that.
    """

    max_split_size_mb: Optional[int] = None
    roundup_power2_divisions: Optional[int] = None
    garbage_collection_threshold: Optional[float] = None
    expandable_segments: bool = False

    @property
    def env(self) -> Dict[str, str]:
        return {"PYTORCH_CUDA_ALLOC_CONF": ",".join(f"{k}:{v}" for k, v in self._asdict().items() if v is not None)}

    def set_env(self) -> None:
        logger.info(f"Setting cuda memory allocator config variable {self.env}")
        os.environ.update(self.env)


CUDA_ALLOCATOR_CONFIG_FOR_DYNAMICALLY_SIZED_DATA = CudaAllocatorConfig(
    roundup_power2_divisions=4, expandable_segments=True
)
