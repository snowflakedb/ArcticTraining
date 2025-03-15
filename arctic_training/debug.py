
# deepspeed or
import torch.distributed as dist
import sys
import torch
import builtins
import fcntl
import gc
import pynvml
import psutil

from deepspeed.accelerator import get_accelerator
from arctic_training.logging import logger

torch_memory_reserved = get_accelerator().memory_reserved
torch_max_memory_reserved = get_accelerator().max_memory_reserved

def exit():
    """useful when one wants to debug dump something and exit cleanly fast"""
    sys.exit()

def gc_empty_accelerator_cache():
    """ runs gc.collect and empties cuda cache.
        this is useful when wanting to test real memory usage
        do not use in production - only during debug - as it can be very expensive
    """
    gc.collect()
    get_accelerator().empty_cache()

def see_memory_usage(message, force=False, ranks=[0]):
    """
    Arguments:
        message: a pre-amble message to print before the counter dumps - useful for annotating where each measurement has been taken - e.g. "before foo" and later "after foo"
        force: allows you to leave see_memory_usage in the code w/o running the code, force=True to activate
        ranks: by default prints only on rank 0 but sometimes we need to debug other ranks, so pass the list like ranks=[1,3]
    """
    #return
    if not force:
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    if not rank in ranks:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # XXX: I think torch.cuda.empty_cache() needs to be called here after gc.collect! (this is the deepspeed version still)
    torch.cuda.empty_cache()

    # collect raw memory usage outside pytorch
    pynvml.nvmlInit()
    rank = dist.get_rank() if dist.is_initialized() else 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    nv_mem = memory_info.used

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)

    accelerator_mem_str = " | ".join([
        f"MA {round(get_accelerator().memory_allocated() / (1024 * 1024 * 1024),2):0.2f} GB",
        f"Max_MA {round(get_accelerator().max_memory_allocated() / (1024 * 1024 * 1024),2):0.2f} GB",
        f"CA {round(torch_memory_reserved() / (1024 * 1024 * 1024),2):0.2f} GB",
        f"Max_CA {round(torch_max_memory_reserved() / (1024 * 1024 * 1024),2):0.2f} GB",
        f"NV {round(nv_mem / (1024 * 1024 * 1024),2):0.2f} GB",
    ])
    cpu_mem_str = f"CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%"

    # add '[rank] mp' prefix to enable easy grep
    print(f"[{rank}] mp: {message}")
    print(f"[{rank}] mp: " + " | ".join([accelerator_mem_str, cpu_mem_str]))

    # get the peak memory to report correct data, so reset the counter for the next call
    get_accelerator().reset_peak_memory_stats()


# fcntl.flock can be slow on shared fs, so if things are too slow especially when many ranks
# are used, you will want it off at a cost of interleaved prints from the same host.
# by default it'll be False to keep things fast, but set it to true when interleaved prints interfere with debug
#
# TODO: alternatively could try to point to some temp file on a local NVME drive - but it's hard to tell if say `/tmp` is on the local drive
USE_PRINTFLOCK = True
#PRINT_FLOCK_FILE = "/tmp/printflock.lock"
PRINT_FLOCK_FILE = __file__

# to quickly temporarily turn off all debugging w/o needing to comment it out - set this to True
# XXX: add API so that the operator could tweak this global from the main script and not mess with this module and commit wrong things by mistake
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

def print_rank(*msg, skip=True, ranks=None):
    """print something on all global ranks with [rank] prefix.
    if `ranks` is passed then only those ranks will be printed

    e.g. to print just on ranks 0 and 3:
    print_rank(*msg, ranks=[0,3]):

    """
    if DISABLE_DEBUG or skip:
        return
    global_rank = dist.get_rank()
    if ranks is not None and global_rank not in ranks:
        return
    print(f"[{global_rank}]", *msg)

def pr(*msg, skip=True, ranks=None):
    """print something on all global ranks with [rank] prefix.
    if `ranks` is passed then only those ranks will be printed

    e.g. to print just on ranks 0 and 3:
    print_rank(*msg, ranks=[0,3]):

    """
    global_rank = dist.get_rank()
    if ranks is not None and global_rank not in ranks:
        return
    print(f"[{global_rank}]", *msg)

def print_rank0(*msg, skip=True):
    if DISABLE_DEBUG or skip:
        return
    """print something only on rank 0"""
    global_rank = dist.get_rank()
    if global_rank == 0:
        print(f"[{global_rank}]", *msg)

def pr0(*msg, skip=True):
    """print something only on rank 0"""
    global_rank = dist.get_rank()
    if global_rank == 0:
        print(f"[{global_rank}]", *msg)


def debug_gathered_tensor(tensor, group, name=None, dim=0):
    """gather a tensor across ranks of the given group and dump its shape and norm


    Arguments:
        tensor: tensor to gather
        group: process group to gather on
        name: optional - the variable name for the tensor
        dim: which dimension to gather on. default: 0

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
    #print_rank0(f"{prefix}:  {gathered_tensor}")
