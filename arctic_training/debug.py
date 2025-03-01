import torch.distributed as dist
import sys
import torch
import builtins
import fcntl

def exit():
    """useful when one wants to debug dump something and exit cleanly fast"""
    sys.exit()

# fcntl.flock can be slow on shared fs, so if things are too slow especially when many ranks
# are used, you will want it off at a cost of interleaved prints from the same host.
# by default it'll be False to keep things fast, but set it to true when interleaved prints interfere with debug
#
# TODO: alternatively could try to point to some temp file on a local NVME drive - but it's hard to tell if say `/tmp` is on the local drive
USE_PRINTFLOCK = True
#PRINT_FLOCK_FILE = "/tmp/printflock.lock"
PRINT_FLOCK_FILE = __file__

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

    from printflock import printflock as print

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
    if skip == True:
        return
    global_rank = dist.get_rank()
    if ranks is not None and global_rank not in ranks:
        return
    print(f"[{global_rank}]", *msg)

def print_rank0(*msg, skip=True):
    if skip == True:
        return
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
