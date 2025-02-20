import torch.distributed as dist
import sys
import torch

def exit():
    """useful when one wants to debug dump something and exit cleanly fast"""
    sys.exit()

def print_rank(*msg):
    """print something on all global ranks with [rank] prefix"""
    global_rank = dist.get_rank()
    print(f"[{global_rank}]", *msg)

def print_rank0(*msg):
    """print something only on rank 0"""
    global_rank = dist.get_rank()
    if global_rank == 0:
        print(f"[{global_rank}]", *msg)


def debug_gathered_tensor(tensor, group, name=None):
    """gather a tensor across ranks of the given group and dump its shape and norm"""

    world_size = dist.get_world_size(group)
    prefix = f"gathered {name}" if name is not None else "gathered"

    tensor = tensor.contiguous()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)

    # concatenate on any dimension since we are just doing norm on everything
    gathered_tensor = torch.cat(tensor_list, dim=0)
    print_rank0(f"{prefix}: shape: {gathered_tensor.shape}")      
    print_rank0(f"{prefix}: norm:  {torch.norm(gathered_tensor)}")      
    #print_rank0(f"{prefix}:  {gathered_tensor}")      
