from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForCausalLM

from .config import LlamaSwiftKVConfig
from .model import LlamaSwiftKVForCausalLM
from .model import LlamaSwiftKVModel


def register_auto():
    AutoConfig.register("llama_swiftkv", LlamaSwiftKVConfig)
    AutoModel.register(LlamaSwiftKVConfig, LlamaSwiftKVModel)
    AutoModelForCausalLM.register(LlamaSwiftKVConfig, LlamaSwiftKVForCausalLM)


__all__ = [
    "LlamaSwiftKVConfig",
    "LlamaSwiftKVForCausalLM",
    "LlamaSwiftKVModel",
    "register_auto",
]
