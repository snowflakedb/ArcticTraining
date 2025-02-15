
import torch.distributed as dist

def print_rank(*msg):
    global_rank = dist.get_rank()
    print(f"[{global_rank}]", *msg)