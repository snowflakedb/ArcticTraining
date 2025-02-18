
import torch.distributed as dist

def print_rank(*msg):
    global_rank = dist.get_rank()
    print(f"[{global_rank}]", *msg)

def print_rank0(*msg):
    global_rank = dist.get_rank()
    if global_rank == 0:
        print(f"[{global_rank}]", *msg)