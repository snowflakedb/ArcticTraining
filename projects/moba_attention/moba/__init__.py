from functools import partial
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .wrapper import moba_layer
from .moba_naive import moba_attn_varlen_naive
from .moba_efficient import moba_attn_varlen
from .moba_with_flash_interface import patch_flash_attn_varlen_func_for_moba
from .config import MoBAConfig


def register_moba(cfg: MoBAConfig):
    ALL_ATTENTION_FUNCTIONS["moba_naive"] = partial(moba_layer, moba_attn_varlen_naive, cfg)
    ALL_ATTENTION_FUNCTIONS["moba"] = partial(moba_layer, moba_attn_varlen, cfg)
