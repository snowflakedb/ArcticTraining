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
from pathlib import Path
import deepspeed.comm as dist


def get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.getenv("WORLD_SIZE", 1))

# delay the local filesystems lookup until it's needed
node_fs_types = None

local_node_fs_types = ["ext", "ext2", "ext3", "ext4", "reiserfs", "jfs", "xfs", "zfs", "xfs", "btrfs", "ntfs", "overlay"]
def is_local_fs(path):
    """ returns True if the `path` resides on the local fs or False otherwise """
    global node_fs_types
    if node_fs_types is None:
        from psutil import disk_partitions
        node_fs_types = {Path(r.mountpoint):r.fstype for r in disk_partitions(all=True)}

    return True if path_to_fs_type(path) in local_node_fs_types else False

def path_to_fs_type(path):
    """
    Given a fs `path` returns the fs type (ext, ext2, etc.) it resides on.
    Note that in this implementation non-existing paths will return the fs type of `/` (which often will be mapped to "overlay")
    This is useful since as long as partitions are mounted already you can detect the type of the fs ven before the sub-dirs were created
    """
    path = Path(path).resolve()
    if path.is_symlink():
        path = path.readlink() # py3.9+

    # assuming at the end we percolate to `/` which is always there so the exit condition is assured
    if path in node_fs_types:
        return node_fs_types[path]

    return path_to_fs_type(path.parent)

def is_main_process_by_path(path):
    if is_local_fs(path):
        return is_local_main_process()
    else:
        return is_global_main_process()

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
def main_process_by_path_first(path):
    """
    Lets the global or the local main process go first inside a with block. The decision which to use is based on the `path`. If the `path` is on a local non-shared fs, we use the local main process. If the path is on the shared fs then it's a global main process.

    The other processes will enter the with block after the defined above main process exits.

    Important: since this context manager uses a barrier it can't be used around code that requires all ranks to work in sync - e.g. gather, barrier, etc. - it'd lead to a deadlock

    Example:

        import time
        with main_process_by_path_first("/shared_fs/cache"):
            # This will be printed first by global process 0 then in a seemingly
            # random order by the other processes.
            # we presume in this example the path is on a shared fs
            global_rank = torch.distributed.get_rank()
            print(f"This will be printed by process {global_rank}")
            time.sleep(5) # emulate actual work
    """
    if is_local_fs(path):
        with _goes_first(is_local_main_process()):
            yield
    else:
        with _goes_first(is_global_main_process()):
            yield

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
    with _goes_first(is_local_main_process()):
        yield


from contextlib import contextmanager


# FlopCounterMode leaks memory via ModuleTracker in `torch<2.6` - fixed in torch-2.6 - if using a lower version switch to a copy of a known good version of torch.utils.module_tracker.ModuleTracker
# BACKCOMPAT: can remove this workaround and the whole `arctic_training/back_compat/module_tracker.py` once we require `torch>=2.6` in dependencies
import torch
from packaging import version
from torch.utils.flop_counter import FlopCounterMode
if version.parse(torch.__version__) < version.parse("2.6"):
    import arctic_training.back_compat.torch.utils.module_tracker
    # override the leaky version with the non-leaky one copied from torch==2.6 - note this only monkey patches `torch.utils.flop_counter`
    torch.utils.flop_counter.ModuleTracker = arctic_training.back_compat.torch.utils.module_tracker.ModuleTracker

class StepFlopCounter:
    """
        This context manager counts floating point operations performed during the execution of the content of the context manager.

        It skips measuring till `start_iter` arrives (usually `start_iter=2` is a good setting - this is because the first iteration always takes more compute than the rest while things are being set up.

        To support variable compute iterations (e.g. when seqlen may vary) it includes an optional caching. As long as the `cache_key` is the same it'll run `FlopCounterMode` once and then will continue returning a cached value. If the `cache_key` hasn't been seen it'll run `FlopCounterMode` once per new key.

        note: flos rather than flops is used in variable names to designate "floating point operations" in order to prevent ambiguity with flops which are "floating point operations per second".

        note: currently there are no docs for `FlopCounterMode` so see notes here: https://github.com/pytorch/pytorch/issues/123800 - one important nuance - it ignores element-wise computations - so it reports less than real, but close enough for large models. Can't beat the convenience.

        Example:

        # don't count for the iteration 1
        step_flos_counter = StepFlopCounter(start_iter=2)
        for iter, batch in enumerate(self.train_batches):
            # cache based on the seqlen of the batch
            with step_flos_counter(iter+1, cache_key=len(batch["input_ids"][0])):
                self.step(batch)
            print(f"{step_flos_counter.get_total_tflos()=}")
    """
    def __init__(self, start_iter=2, display=False):
        self.target_iter = start_iter
        self.flos_counter = FlopCounterMode(display=display)
        self.flos = 0
        self.cache = {}

    @contextmanager
    def __call__(self, iter, cache_key=0):
        """
            Arguments:
                iter: current iteration id counting from 1
                cache_key: whatever value you want to cache on - e.g. seqlen

        """

#        if iter in [2,3,4,5,6]:
        if iter > 1:
            with self.flos_counter:
                yield
            self.flos = self.flos_counter.get_total_flops()
            return
        else:
            yield
            return

        # skip first steps while things are unstable
        if iter < self.target_iter:
            yield
            return

        # avoid recalculating and pull from cache instead
        if cache_key in self.cache:
            print(f"Cache hit for {iter=} {cache_key=}")
            self.flos = self.cache[cache_key]
            yield
            return

        with self.flos_counter:
            yield
        self.flos = self.flos_counter.get_total_flops()
        self.cache[cache_key] = self.flos

    def get_total_tflos(self):
        return self.flos / 1e12

import torch
def gather_sum_number(number, device, group=None):
    tensor = torch.tensor(number).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor.item()

def gather_mean_number(number, device, group=None):
    tensor = torch.tensor(number).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MEAN, group=group)
    return tensor.item()

def gather_sum_tensor(tensor, device, group=None):
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return tensor

def gather_mean_tensor(tensor, device, group=None):
    dist.all_reduce(t, op=dist.ReduceOp.MEAN, group=group)
    return tensor

def gather_object(number, device, group=None):
    """ returns a list of objects """
    # XXX: which world size? probably better to always specify the group explicitly and derive from it to avoid bugs and assumptions
    output = [None for _ in range(get_world_size())]
    torch.distributed.all_gather_object(output, number, group=group)
    return output

def format_human_base2_number(num, suffix="B"):
    """
    formats base-2 numbers to human readable format, e.g. format_number(10000001) => '9.5MiB'
    """
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
