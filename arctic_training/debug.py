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


import builtins
import fcntl
import gc
import os

import psutil
import torch
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator

can_run_pynvml = True
try:
    import pynvml

    pynvml.nvmlInit()
except Exception:
    can_run_pynvml = False

torch_memory_reserved = get_accelerator().memory_reserved
torch_max_memory_reserved = get_accelerator().max_memory_reserved

pynvml_handle = None


def get_local_rank():
    return int(os.getenv("LOCAL_RANK", 0))


def get_nvml_mem():
    global pynvml_handle

    if not can_run_pynvml:
        return 0

    if pynvml_handle is None:
        rank = get_local_rank() if dist.is_initialized() else 0
        pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
        # pynvml.nvmlShutdown()
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(pynvml_handle)
    return memory_info.used


def gc_empty_accelerator_cache():
    """runs gc.collect and empties cuda cache.
    this is useful when wanting to test real memory usage
    do not use in production - only during debug - as it can be very expensive
    """
    gc.collect()
    get_accelerator().empty_cache()


def see_memory_usage(message, force=False, ranks=[0]):
    """
    Arguments:
        - `message`: a pre-amble message to print before the counter dumps - useful for annotating where each measurement has been taken - e.g. "before foo" and later "after foo"
        - `force`: allows you to leave see_memory_usage in the code w/o running the code, force=True to activate
        - `ranks`: by default prints only on rank 0 but sometimes we need to debug other ranks, so pass the list like ranks=[1,3]
    """
    if not force:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank not in ranks:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # In some situations we want to flush the cache but not others, so for now let the developer
    # override this manually - by default it should not be called. when it's not enabled use the
    # MA_* numbers to get the real memory usage, rather than CA_* ones
    # torch.cuda.empty_cache()

    # collect raw memory usage outside pytorch
    nv_mem = get_nvml_mem()

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)

    accelerator_mem_str = " | ".join(
        [
            f"MA {round(get_accelerator().memory_allocated() / 2**30, 2):0.2f} GB",
            f"Max_MA {round(get_accelerator().max_memory_allocated() / 2**30, 2):0.2f} GB",
            f"CA {round(torch_memory_reserved() / 2**30, 2):0.2f} GB",
            f"Max_CA {round(torch_max_memory_reserved() / 2**30, 2):0.2f} GB",
            f"NV {round(nv_mem / 2**30, 2):0.2f} GB",
        ]
    )
    cpu_mem_str = f"CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%"

    # add '[rank] mp' prefix to enable easy grep
    print(f"[{rank}] mp: {message}")
    print(f"[{rank}] mp: " + " | ".join([accelerator_mem_str, cpu_mem_str]))

    # get the peak memory to report correct data, so reset the counter for the next call
    get_accelerator().reset_peak_memory_stats()


def get_mem_metrics():

    gc.collect()
    # torch.cuda.empty_cache()

    nv_mem = get_nvml_mem()

    summary = " | ".join(
        [
            f"MA {round(get_accelerator().memory_allocated() / 2**30, 2):0.2f} GB",
            f"Max_MA {round(get_accelerator().max_memory_allocated() / 2**30, 2):0.2f} GB",
            f"NV {round(nv_mem / 2**30, 2):0.2f} GB",
        ]
    )

    # get the peak memory to report correct data, so reset the counter for the next call
    # this will lead to wrong peak reports if `see_mem_usage` is also used during the run,
    # as it resets the peak counter and there is only one counter
    get_accelerator().reset_peak_memory_stats()

    return summary


# fcntl.flock can be slow on shared fs, so if things are too slow especially when many ranks are
# used, you will want it off at a cost of interleaved prints from the same host. by default it'll
# be False to keep things fast, but set it to true when interleaved prints interfere with debug
#
# TODO: alternatively could try to point to some temp file on a local NVME drive - but it's hard to
# tell if say `/tmp` is on the local drive
USE_PRINTFLOCK = True
# PRINT_FLOCK_FILE = "/tmp/printflock.lock"
PRINT_FLOCK_FILE = __file__

# Set to True to quickly temporarily turn off all debugging w/o needing to disable each call
#
# XXX: perhaps add API so that the operator could tweak this global from the main script and not
# mess with this module and commit wrong things by mistake
DISABLE_DEBUG = True


def printflock(*args, **kwargs):
    """
    This is a wrapper around the built-in Python `print` which calls `flock` before calling
    `print` and unlocks it immediately after. This wrapper is useful for when each rank needs to
    print a message without getting it interleaved with prints from other ranks.
    The lock file is the file this wrapper is defined in.
    The output order will be random per rank.

    Example:
        >>> # assuming 4 GPUs
        >>> world_size = dist.get_world_size()
        >>> rank = dist.get_rank()
        >>> printflock(f"This is a very long message from rank {rank}/{world_size}")
       This is a very long message from rank 0/4
       This is a very long message from rank 2/4
       This is a very long message from rank 3/4
       This is a very long message from rank 1/4

    It can also be used to override normal `print`:

    from arctictraining.debug import printflock as print

    and then you don't need to change anything in your code.
    """

    #    with open(__file__, "r") as fh:
    with open(PRINT_FLOCK_FILE, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


if USE_PRINTFLOCK:
    print = printflock


def print_rank(*msg, force=False, ranks=None):
    """print something on all global ranks with [rank] prefix.
    if `ranks` is passed then only those ranks will be printed

    e.g. to print just on ranks 0 and 3:
    print_rank(*msg, ranks=[0,3]):

    """
    if DISABLE_DEBUG or not force or ranks is None:
        return
    global_rank = dist.get_rank()
    if global_rank not in ranks:
        return
    print(f"[{global_rank}]", *msg)


def print_rank0(*msg, force=False):
    """print something only on rank 0"""
    if DISABLE_DEBUG or not force:
        return
    global_rank = dist.get_rank()
    if global_rank == 0:
        print(f"[{global_rank}]", *msg)


def debug_gathered_tensor(tensor, group, name=None, dim=0):
    """gather a tensor across ranks of the given group and dump its shape and norm

    Arguments:
      - `tensor`: tensor to gather
      - `group`: process group to gather on
      - `name`: optional - the variable name for the tensor
      - `dim`: which dimension to gather on. default: 0

    """

    world_size = dist.get_world_size(group)
    prefix = f"gathered {name}" if name is not None else "gathered"

    tensor = tensor.contiguous()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)

    # concatenate on any dimension since we are just doing norm on everything
    gathered_tensor = torch.cat(tensor_list, dim=dim)
    print_rank0(f"{prefix}: shape: {gathered_tensor.shape}")
    print_rank0(f"{prefix}: norm:  {torch.norm(gathered_tensor)}")
    # print_rank0(f"{prefix}:  {gathered_tensor}")
