import deepspeed
import torch
from arctic_training.config import DataConfig
from arctic_training.data import data_factory

device = torch.device("cuda:0")
deepspeed.init_distributed()

config = DataConfig(
    tokenizer="meta-llama/Meta-Llama-3.1-8B-Instruct",
    datasets=["ise-uiuc/Magicoder-OSS-Instruct-75K"],
    use_data_cache=True,
    data_cache_dir="/data-fast/data/",
    num_proc=1,
)
data, eval = data_factory(None, config)
print(type(data))
