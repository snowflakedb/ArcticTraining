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
from contextlib import contextmanager
import deepspeed.comm as dist


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", 1))

def is_local_main_process():
    return get_local_rank() == 0

def is_global_main_process():
    return dist.get_rank() == 0

@contextmanager
def _goes_first(is_main: bool):
    if not is_main:
        dist.barrier()

    yield

    if is_main:
        dist.barrier()

@contextmanager
def global_main_process_first():
    """
    Lets the global main process go first inside a with block.

    The other processes will enter the with block after the global main process exits.

    Important: since this context manager uses a barrier it can't be used around code that requires all ranks to work in sync - e.g. gather, barrier, etc. - it'd lead to a deadlock

    Example:

        import time
        global_rank = torch.distributed.get_rank()
        with global_main_process_first():
            # This will be printed first by global process 0 then in a seemingly
            # random order by the other processes.
            print(f"This will be printed by process {global_rank}")
            time.sleep(5) # emulate actual work
    """
    with _goes_first(is_global_main_process()):
        yield

@contextmanager
def local_main_process_first():
    """
    Lets the local main process go inside a with block.

    The other processes will enter the with block after the local main process exits.

    Important: since this context manager uses a barrier it can't be used around code that requires all ranks to work in sync - e.g. gather, barrier, etc. - it'd lead to a deadlock

    Example:

        import time
        local_rank = get_local_rank()
        with local_main_process_first():
            # This will be printed first by local process 0 then in a seemingly
            # random order by the other processes.
            print(f"This will be printed by process {local_rank}")
            time.sleep(5) # emulate actual work
    """
    local_rank = get_local_rank()
    with _goes_first(is_local_main_process()):
        yield

