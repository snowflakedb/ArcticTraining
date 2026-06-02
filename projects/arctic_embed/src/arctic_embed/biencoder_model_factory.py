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

from typing import Any
from typing import Dict
from typing import cast

from peft import get_peft_model
from peft.config import PeftConfig
from transformers import AutoConfig
from transformers import AutoModel

from arctic_training.config.model import ModelConfig
from arctic_training.model.factory import ModelFactory
from arctic_training.model.hf_factory import HFModelFactory

from .core.biencoder_model import Biencoder
from .core.biencoder_model import PoolingOption


class BiencoderModelConfig(ModelConfig):
    type: str = "biencoder"
    pooling: PoolingOption = "first_token"
    kwargs: Dict[str, Any] = {}
    # Gradient/activation checkpointing granularity: checkpoint every n-th layer.
    # 1 = every layer (max memory savings, max recompute); higher = less recompute.
    activation_checkpoint_every_n: int = 1
    # Leave the last k transformer layers un-checkpointed (full activations kept).
    # Composes with every_n. Use spare VRAM to skip recompute of the final layers.
    activation_checkpoint_uncheckpointed_last_k: int = 0
    # torch.compile the encoder after construction. Off by default: the per-batch
    # variable sequence length can trigger recompiles that erase the speedup.
    torch_compile: bool = False


class BiencoderModelFactory(ModelFactory):
    """A Biencoder-specific HuggingFace model factory.

    NOTE: This is similar to the HuggingFace HFModelFactory, but it uses `AutoModel`
    instead of `AutoModelForCausalLM` and wraps the result into a `Biencoder`.
    """

    name = "biencoder"
    config: BiencoderModelConfig

    def create_config(self):
        arctic_training_model_config = self.config
        assert isinstance(arctic_training_model_config, BiencoderModelConfig)
        return AutoConfig.from_pretrained(self.config.name_or_path, **arctic_training_model_config.kwargs)

    def create_model(self, model_config: AutoConfig) -> Biencoder:
        arctic_training_model_config = self.config
        assert isinstance(arctic_training_model_config, BiencoderModelConfig)
        trust_remote_code = arctic_training_model_config.kwargs.get("trust_remote_code", None)
        encoder = AutoModel.from_pretrained(
            self.config.name_or_path,
            config=model_config,
            attn_implementation=self.config.attn_implementation,
            torch_dtype=self.config.dtype.value,
            trust_remote_code=trust_remote_code,
        )
        # Train Qwen3 as a bidirectional encoder: disable causal masking on every
        # attention layer. The matching non-causal attention bias is supplied in
        # `Biencoder.encode` (see `_qwen3_attention_masks`).
        if getattr(model_config, "model_type", "") == "qwen3":
            for layer in encoder.layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "is_causal"):
                    layer.self_attn.is_causal = False
        return Biencoder(encoder, pooling=arctic_training_model_config.pooling)

    def post_create_model_callback(self, model: Biencoder):
        if self.config.peft_config:
            # NOTE: This typecast is technically incorrect but should work in practice.
            peft_config = cast(PeftConfig, self.config.peft_config)
            model.encoder = get_peft_model(model.encoder, peft_config)

        if not self.config.disable_activation_checkpoint:
            import logging

            import torch

            n = self.config.activation_checkpoint_every_n
            last_k = getattr(self.config, "activation_checkpoint_uncheckpointed_last_k", 0)
            if n < 1:
                raise ValueError(f"activation_checkpoint_every_n must be >= 1, got {n}")

            model.encoder = HFModelFactory.make_model_gradient_checkpointing_compatible(model.encoder)
            # transformers sets `_gradient_checkpointing_func` AND
            # `gradient_checkpointing=True` on every GradientCheckpointingLayer here
            # (use_reentrant=False is the recommended, non-deprecated mode). We then
            # selectively disable it on some layers below.
            model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model.encoder, "config") and hasattr(model.encoder.config, "use_cache"):
                model.encoder.config.use_cache = False

            if n > 1 or last_k > 0:
                layers = getattr(model.encoder, "layers", None)
                if layers is None:
                    # AutoModel may nest the layer stack; find the decoder ModuleList.
                    for module in model.encoder.modules():
                        if (
                            isinstance(module, torch.nn.ModuleList)
                            and len(module) > 0
                            and hasattr(module[0], "gradient_checkpointing")
                        ):
                            layers = module
                            break
                if layers is None:
                    raise ValueError("Could not locate the transformer layer stack for gradient checkpointing.")
                num_layers = len(layers)
                kept = 0
                for i, layer in enumerate(layers):
                    # Checkpoint every n-th layer, but never the last `last_k` layers:
                    # leaving the final layers un-checkpointed spends spare memory to
                    # skip their recompute (cheapest recompute to drop — needed first
                    # in backward). `last_k` lets us tune recompute to fill VRAM headroom.
                    ckpt = (i % n == 0) and (i < num_layers - last_k)
                    layer.gradient_checkpointing = ckpt
                    kept += int(ckpt)
                logging.getLogger(__name__).info(
                    f"Gradient checkpointing on {kept}/{num_layers} layers (every {n}, last {last_k} uncheckpointed)."
                )

        if getattr(self.config, "torch_compile", False):
            import logging

            import torch

            # dynamic=True avoids a recompile per distinct (batch, seqlen) shape,
            # which is essential here: every batch has a different padded length.
            model.encoder = torch.compile(model.encoder, dynamic=True)
            logging.getLogger(__name__).info("torch.compile enabled on encoder (dynamic=True).")

        return model
