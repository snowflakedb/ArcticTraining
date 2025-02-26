import torch.distributed as dist
import sys
import torch

def exit():
    """useful when one wants to debug dump something and exit cleanly fast"""
    sys.exit()

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
