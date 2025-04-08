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

# This file was original taken from the following:
# https://github.com/MoonshotAI/MoBA/tree/61e456bc956c5a25fd9c84e5496b661329cb1b72
# Modification may have been made by Snowflake

from functools import partial

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .config import MoBAConfig
from .moba_efficient import moba_attn_varlen
from .moba_naive import moba_attn_varlen_naive
from .moba_with_flash_interface import patch_flash_attn_varlen_func_for_moba
from .wrapper import moba_layer


def register_moba(cfg: MoBAConfig):
    ALL_ATTENTION_FUNCTIONS["moba_naive"] = partial(moba_layer, moba_attn_varlen_naive, cfg)
    ALL_ATTENTION_FUNCTIONS["moba"] = partial(moba_layer, moba_attn_varlen, cfg)
