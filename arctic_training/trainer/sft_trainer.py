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

import math
from typing import Union

import torch
import torch.distributed.nn.functional
from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledFusedLogitsLoss

from arctic_training.checkpoint.ds_engine import DSCheckpointEngine
from arctic_training.checkpoint.hf_engine import HFCheckpointEngine
from arctic_training.data.sft_factory import SFTDataFactory
from arctic_training.logging import logger
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.model.liger_factory import LigerModelFactory
from arctic_training.optimizer.adam_factory import CPUAdamOptimizerFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.scheduler.hf_factory import HFSchedulerFactory
from arctic_training.tokenizer.hf_factory import HFTokenizerFactory
from arctic_training.trainer.trainer import Trainer
from arctic_training.trainer.utils import to_device

# Warning message template for zero trainable tokens
ZERO_TOKENS_WARNING = (
    "Batch has no trainable tokens {} (all labels are -100). "
    "Returning zero loss. This may indicate data issues with too many "
    "empty/masked outputs or an unfavorable packing distribution."
)


class SFTTrainer(Trainer):
    name = "sft"
    data_factory: SFTDataFactory
    model_factory: Union[HFModelFactory, LigerModelFactory]
    checkpoint_engine: Union[DSCheckpointEngine, HFCheckpointEngine]
    optimizer_factory: Union[FusedAdamOptimizerFactory, CPUAdamOptimizerFactory]
    scheduler_factory: Union[HFSchedulerFactory]
    tokenizer_factory: Union[HFTokenizerFactory]

    @staticmethod
    def _count_trainable_tokens(labels: torch.Tensor) -> torch.Tensor:
        """Count non-masked tokens in labels tensor.

        Args:
            labels: Labels tensor with -100 for masked positions

        Returns:
            Scalar tensor with count of non-masked tokens
        """
        return ((labels != -100).view(-1)).sum()

    def _compute_sample_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        packed_sample_seqlens: list,
    ) -> torch.Tensor:
        """
        Compute sample-weighted loss for packed sequences.

        Instead of averaging loss over all tokens (which biases toward longer responses),
        this computes the loss for each sample separately and averages the per-sample losses.
        This ensures each sample contributes equally regardless of response length.

        Args:
            logits: Model logits, shape (batch_size, seq_len, vocab_size)
            labels: Labels with -100 for masked positions, shape (batch_size, seq_len)
            packed_sample_seqlens: List of lists, each inner list contains sequence lengths
                                   of samples packed into that batch element

        Returns:
            Scalar loss tensor (average of per-sample losses)
        """
        import torch.nn.functional as F

        batch_size = logits.shape[0]
        # Use float32 for loss accumulation to match F.cross_entropy behavior
        # (cross_entropy internally upcasts to float32 for numerical stability)
        total_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        total_samples = 0

        for batch_idx in range(batch_size):
            sample_seqlens = packed_sample_seqlens[batch_idx]
            pos = 0

            for sample_len in sample_seqlens:
                # Extract this sample's logits and labels
                sample_logits = logits[batch_idx, pos : pos + sample_len - 1]  # shift for causal LM
                sample_labels = labels[batch_idx, pos + 1 : pos + sample_len]  # shift for causal LM

                # Count trainable tokens in this sample
                mask = sample_labels != -100
                num_trainable = mask.sum()

                if num_trainable > 0:
                    # Compute loss for this sample (mean over its trainable tokens)
                    sample_loss = F.cross_entropy(
                        sample_logits.reshape(-1, sample_logits.shape[-1]),
                        sample_labels.reshape(-1),
                        ignore_index=-100,
                        reduction="mean",
                    )
                    total_loss = total_loss + sample_loss
                    total_samples += 1

                pos += sample_len

        # Average over samples (not tokens!)
        if total_samples > 0:
            return total_loss / total_samples
        else:
            # No trainable samples - return zero loss connected to logits for gradient flow
            # Cast to float32 for consistency with F.cross_entropy output dtype
            return (logits.sum() * 0.0).to(torch.float32)

    def loss(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)

        if self.config.sequence_parallel_size == 1:
            # Check if we have packed samples and need sample-weighted loss
            packed_sample_seqlens = batch.pop("packed_sample_seqlens", None)
            use_sample_weighted_loss = (
                packed_sample_seqlens is not None
                and getattr(self.config.data, "pack_samples", False)
                and getattr(self.config.data, "sample_weighted_loss", False)
            )

            # For sample-weighted loss, we need access to logits to compute per-sample losses.
            # With Liger models, calling model(**batch) with labels triggers fused loss computation
            # which doesn't return logits. To get logits, we must call the model WITHOUT labels.
            # This bypasses Liger's memory-efficient fused loss but enables sample weighting.
            if use_sample_weighted_loss:
                # Pop labels from batch - we'll compute loss manually
                labels = batch.pop("labels")

                # Log warning about memory trade-off (only once per training run)
                if not hasattr(self, "_sample_weighted_loss_warned"):
                    self._sample_weighted_loss_warned = True
                    if self.config.model.type == "liger":
                        logger.warning(
                            "sample_weighted_loss=True with Liger model: bypassing fused cross-entropy "
                            "to access logits. This uses more GPU memory than default Liger behavior. "
                            "Each sample will contribute equally to the gradient regardless of length."
                        )
                    else:
                        logger.info(
                            "sample_weighted_loss=True: each sample contributes equally to the gradient "
                            "regardless of response length."
                        )

                # Call model WITHOUT labels to get logits (works with both Liger and HF models)
                outputs = self.model(**batch, use_cache=False)

                # Defensive check: ensure logits are available
                if not hasattr(outputs, "logits") or outputs.logits is None:
                    raise RuntimeError(
                        "sample_weighted_loss=True requires access to model logits, but the model "
                        "returned None or missing logits. This can happen if: (1) the model architecture "
                        "doesn't support returning logits, or (2) labels were accidentally passed to the "
                        "model. Ensure labels are removed from the batch before calling the model."
                    )
                logits = outputs.logits

                # Handle zero-token cases
                good_tokens = self._count_trainable_tokens(labels)
                if good_tokens.item() == 0:
                    logger.warning(ZERO_TOKENS_WARNING.format(""))
                    # Maintain gradient connectivity; cast to float32 for consistency with
                    # F.cross_entropy which internally upcasts for numerical stability
                    loss = (logits.sum() * 0.0).to(torch.float32)
                else:
                    # Compute sample-weighted loss
                    loss = self._compute_sample_weighted_loss(logits, labels, packed_sample_seqlens)

                return loss

            # Default path: use HF/Liger's built-in loss computation
            # if model.type=liger is configured - this will use a much more efficient fused
            # logits+loss liger kernel - using significantly less gpu memory and a bit faster
            # compute (liger fused logits+loss kernel does not repeat forward during backward)
            outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss

            # Handle zero-token cases to avoid NaN loss and ensure consistent behavior.
            # IMPORTANT: We must maintain gradient connectivity to avoid NCCL hangs.
            # Creating a disconnected tensor (torch.tensor(0.0)) would cause this rank
            # to skip gradient computation while other ranks wait in ALLREDUCE.
            labels = batch.get("labels")
            if labels is not None:
                good_tokens = self._count_trainable_tokens(labels)
                if good_tokens.item() == 0:
                    logger.warning(ZERO_TOKENS_WARNING.format(""))
                    # Maintain gradient connectivity by using a tensor connected to model outputs.
                    # With Liger, outputs.logits is None, so we check alternatives.
                    if hasattr(outputs, "logits") and outputs.logits is not None:
                        # Use logits to maintain gradient flow (same pattern as tiled compute path)
                        loss = (outputs.logits.sum() * 0.0).to(loss.dtype)
                    elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                        # Use hidden states if available
                        loss = (outputs.hidden_states[-1].sum() * 0.0).to(loss.dtype)
                    else:
                        # For Liger: loss IS connected to the graph, but may be NaN.
                        # We can't use nan_to_num as it breaks gradient flow.
                        # Instead, check if loss is valid - if so, multiply by 0.
                        # If NaN/Inf, we need to do a dummy forward to get a connected tensor.
                        if not (torch.isnan(loss) or torch.isinf(loss)):
                            loss = loss * 0.0
                        else:
                            # NaN case: we need gradients to flow but can't use NaN * 0 = NaN.
                            # Access the model's first parameter to create a connected zero.
                            # This ensures gradient buffers are allocated for all parameters.
                            first_param = next(self.model.parameters())
                            loss = (first_param.sum() * 0.0).to(loss.dtype)
                elif torch.isnan(loss) or torch.isinf(loss):
                    # Has good tokens but NaN/Inf loss - this is a real numerical issue, let it propagate
                    pass

            return loss

        # Ulysses SP expectations:
        # 1. batch has `labels`` replaced with `shift_labels`` (which are already preshifted in
        #    DataLoader)
        # 2. this rank deals with a seqlen dimension shard so once the loss is calculated it needs
        #    to do a differentiable weighted loss average to get the grads right

        if "labels" in batch:
            raise ValueError(
                "found labels in batch - they shouldn't be there, instead shift_labels should be there - check"
                " that UlyssesSPDataLoaderAdapter has been applied to the original DataLoader object"
            )
        if "shift_labels" not in batch:
            raise ValueError(
                "shift_labels are missing from the batch - check that UlyssesSPDataLoaderAdapter has been"
                " applied to the original DataLoader object"
            )

        shift_labels = batch["shift_labels"]

        # We have 2 implementation of efficient tiled logits+loss computation.
        # 1. Liger fused cross-entropy is the fastest/most memory efficient way - liger-kernel
        #    doesn't recompute forward inside backward, instead it computes the gradients in the
        #    forward path.
        # 2. But liger kernel isn't implemented for all HF Transformers models, so then we fall
        #    back onto our tiled logits+loss compute implementation that is almost as efficient
        #    memory-wise, but which has more compute overhead before backward re-runs forward. The
        #    total memory usage is very similar, but cuda cache flushes earlier if pushing close to
        #    OOM than liger.
        # Note: When in eval mode with SP > 1, liger's fused cross-entropy returns None, so we
        # fall back to tiled compute for evaluation even when model.type == "liger".
        # Use model_unwrapped.training because self.model is a DeepSpeed engine wrapper.
        use_liger_fused_loss = self.config.model.type == "liger" and self.model_unwrapped.training

        if use_liger_fused_loss:

            # letting liger do fused logits+loss calculation
            outputs = self.model(**batch, use_cache=False)
            loss = outputs.loss

            if loss is None:
                raise ValueError(
                    "Liger-Kernel failed to compute loss (returned None). This is unexpected during training."
                )

            # Handle zero-token and NaN cases from Liger.
            # IMPORTANT: Must maintain gradient connectivity to avoid NCCL hangs.
            good_tokens = self._count_trainable_tokens(shift_labels)
            if good_tokens.item() == 0:
                logger.warning(ZERO_TOKENS_WARNING.format("on this SP rank"))
                # Maintain gradient connectivity - check if loss is valid first
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss = loss * 0.0
                else:
                    # NaN case: use model parameter to maintain gradient flow
                    first_param = next(self.model.parameters())
                    loss = (first_param.sum() * 0.0).to(loss.dtype)
            elif torch.isnan(loss) or torch.isinf(loss):
                # Has good tokens but NaN/Inf loss - real numerical issue, let it propagate
                pass

        else:
            # Currently relying on an automatic num_shards derivation based on the goal that it'll
            # take approximately 1GB of fp32 logits in a shard, could make this configurable if
            # desired later. Less than 1GB doesn't seem to make much of an impact, but perhaps a
            # higher number will be more efficient as it'll run less shards.
            num_shards = "auto"
            if num_shards == "auto":
                # parameterize to about 1GB fp32 logits shards
                slice_size_in_gb = 1  # XXX: make configurable?
                bs, seqlen = shift_labels.shape
                vocab_size = self.model_unwrapped.config.vocab_size
                logits_numel = bs * seqlen * vocab_size
                size_in_gb = logits_numel * 4 / 2**30  # fp32
                # the sp shard's seqlen sp shard can be easily not divisible by the derived number
                # of chunked loss shards, so we use the uppper ceiling and allow the last chunk to
                # be shorter than the rest
                num_shards = math.ceil(size_in_gb / slice_size_in_gb)
                # print(f"derived {num_shards} shards for size {size_in_gb}GB")

            model_with_head = self.model_unwrapped
            outputs = model_with_head.model(**batch, use_cache=False)
            hidden_states = outputs.last_hidden_state
            compute_params = [model_with_head.lm_head.weight]
            seqlen = shift_labels.shape[1]
            mask = None
            output_reduction = "sum"

            # since -100s shift_labels are ignored we have to perform a weighted average on each
            # loss slice as each slice may contribute a different number of non- -100 labels
            def fused_logits_loss_fn(model_with_head=None, hidden_states=None, shift_labels=None):
                vocab_size = model_with_head.config.vocab_size
                logits = model_with_head.lm_head(hidden_states)
                if (shift_labels == -100).all():
                    # fake loss calculation, since CE will return nan, but grads will be set
                    # a normal loss_fn upcasts logits to float so match it
                    loss_sum = (logits.sum() * 0.0).float()
                else:
                    good_items = self._count_trainable_tokens(shift_labels)
                    loss = model_with_head.loss_function(
                        logits=logits, labels=None, vocab_size=vocab_size, shift_labels=shift_labels
                    )
                    loss_sum = loss * good_items
                return loss_sum

            total_loss_sum = TiledFusedLogitsLoss.apply(
                fused_logits_loss_fn,
                model_with_head,
                hidden_states,
                shift_labels,
                mask,
                num_shards,
                compute_params,
                output_reduction,
            )
            total_good_items = self._count_trainable_tokens(shift_labels)

            # Handle NaN/Inf and zero-token cases from tiled compute.
            # Note: fused_logits_loss_fn already handles all-masked batches with (logits.sum() * 0.0),
            # so total_loss_sum should be connected to the graph even for all-masked batches.
            if total_good_items.item() == 0:
                logger.warning(ZERO_TOKENS_WARNING.format("on this rank"))
                # total_loss_sum is connected to the graph via fused_logits_loss_fn's logits.sum() * 0.0
                # Just ensure we return a proper zero (multiply to handle any edge cases)
                loss = (
                    total_loss_sum * 0.0
                    if not (torch.isnan(total_loss_sum) or torch.isinf(total_loss_sum))
                    else (hidden_states.sum() * 0.0).to(total_loss_sum.dtype)
                )
            elif torch.isnan(total_loss_sum) or torch.isinf(total_loss_sum):
                # Has good tokens but NaN/Inf loss - real numerical issue, let it propagate
                loss = total_loss_sum / total_good_items
            else:
                # Normal case
                loss = total_loss_sum / total_good_items

        # differentiable weighted per-shard-loss aggregation across ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=self.sp_group)
        good_tokens = self._count_trainable_tokens(shift_labels)
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(good_tokens, group=self.sp_group)
        total_loss = sum(losses_per_rank[rank] * good_tokens_per_rank[rank] for rank in range(self.sp_world_size))
        total_good_tokens = sum(good_tokens_per_rank)

        # Protect against division by zero when all tokens are masked
        # This can happen with packed samples that have mostly non-assistant content
        if total_good_tokens.item() == 0:
            logger.warning(ZERO_TOKENS_WARNING.format("across all SP ranks"))
            # Maintain gradient connectivity by using the per-rank loss (which is connected)
            # rather than creating a disconnected tensor
            loss = loss * 0.0
        else:
            loss = total_loss / total_good_tokens

        return loss
