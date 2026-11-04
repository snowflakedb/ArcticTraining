# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed-sample boundaries for linear-attention (GDN) mixers.

A packed SFT batch carries only ``position_ids``. Full-attention layers recover sample
boundaries from them inside ``_flash_attention_forward``, but Qwen3.5's GDN mixer
(``linear_attn``) only resets its recurrent state at ``cu_seq_lens_q``, and its short conv
only stops at ``seq_idx``; neither is derived from ``position_ids`` by the model. Without
them every GDN layer runs the whole packed row as one sequence, so state and the conv window
carry from one sample into the next.

``install_packed_gdn_boundaries`` adds a top-level forward pre-hook that derives
``cu_seq_lens_q/k``, ``max_length_q/k`` and ``seq_idx`` from ``position_ids`` and passes them
down as model kwargs. Calls that already carry ``cu_seq_lens_q`` are left alone.
"""

import torch


def model_has_linear_attn(model) -> bool:
    return any(hasattr(m, "linear_attn") for m in model.modules())


def install_packed_gdn_boundaries(model) -> bool:
    """Install the boundary pre-hook on ``model``; returns False if it has no GDN mixers."""
    if getattr(model, "_at_packed_gdn_boundaries_installed", False):
        return True
    if not model_has_linear_attn(model):
        return False

    from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids

    def _pre_hook(module, args, kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is None or "cu_seq_lens_q" in kwargs:
            return None
        if position_ids.dim() != 2 or position_ids.shape[0] != 1:
            raise RuntimeError(
                "packed GDN boundaries need position_ids of shape [1, seqlen], got" f" {tuple(position_ids.shape)}"
            )
        (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(position_ids)
        lengths = cu_q.diff().long()
        seq_idx = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=cu_q.device, dtype=torch.int32), lengths
        ).unsqueeze(0)
        kwargs = dict(kwargs)
        kwargs.update(cu_seq_lens_q=cu_q, cu_seq_lens_k=cu_k, max_length_q=max_q, max_length_k=max_k, seq_idx=seq_idx)
        return args, kwargs

    model.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    model._at_packed_gdn_boundaries_installed = True
    return True
