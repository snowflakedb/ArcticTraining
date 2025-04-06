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

import torch
from moba import MoBAConfig
from moba import register_moba
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--moba-chunk-size", type=int, default=4096)
    parser.add_argument("--moba-topk", type=int, default=12)
    parser.add_argument(
        "--attn",
        default="moba",
        help="choose attention backend",
        choices=["flash_attention_2", "moba", "moba_naive"],
    )
    args = parser.parse_args()

    register_moba(MoBAConfig(args.moba_chunk_size, args.moba_topk))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation=args.attn,
    )
    tknz = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompt = "how are you?"
    input_tokens = tknz.encode(prompt)
    input_ids = torch.tensor([input_tokens], device=model.device)
    tokens = model.generate(input_ids, max_length=32, do_sample=False)
    print(tokens)
    print(tknz.decode(tokens.squeeze().tolist()))
