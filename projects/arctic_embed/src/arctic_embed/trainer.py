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

from __future__ import annotations

import os
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import deepspeed.comm as dist
import torch
import torch.nn.functional as F
from deepspeed import DeepSpeedEngine
from torch import Tensor
from tqdm.auto import tqdm

from arctic_training.config.trainer import TrainerConfig
from arctic_training.logging import logger
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.scheduler.hf_factory import HFSchedulerFactory
from arctic_training.scheduler.wsd_factory import WSDSchedulerFactory
from arctic_training.tokenizer.factory import TokenizerFactory
from arctic_training.trainer.trainer import Trainer

from .biencoder_model_factory import BiencoderModelConfig
from .biencoder_model_factory import BiencoderModelFactory

# from .checkpointing import BiencoderCheckpointEngine
from .biencoder_s3_checkpoint import BiencoderS3CheckpointEngine
from .contrastive_dataloader import ContrastivePretokenizedDataConfig
from .contrastive_dataloader import ContrastivePretokenizedDataFactory
from .core.biencoder_model import Biencoder
from .core.losses import info_nce_loss
from .core.losses import one_size_truncated_mrl_info_nce_loss
from .core.pretokenized_batch_loader import ContrastiveLearningBatch


class BiencoderTrainerConfig(TrainerConfig):
    type: str = "biencoder"
    use_in_batch_negatives: bool = False
    loss_temperature: float = 0.02
    model: BiencoderModelConfig
    data: ContrastivePretokenizedDataConfig
    mrl_dim: Optional[int] = None
    eval_interval: Optional[int] = None
    # SPLADE regularization (v1 L1 is deprecated; use FLOPs per side instead)
    splade_reg_weight: float = 0.0  # deprecated; kept for backward-compat log
    splade_flops_weight_query: float = 0.0
    splade_flops_weight_doc: float = 0.0
    splade_nnz_threshold: float = 1e-3


class FakeTokenizer:
    def save_pretrained(self, *args, **kwargs):
        pass


class FakeTokenizerFactory(TokenizerFactory):
    name = "fake"

    def create_tokenizer(self):
        return FakeTokenizer()


def rescale_grad_cb(self: Trainer) -> None:
    """Rescale the gradients to account for the fact that we're averaging gradients
    during the backwards pass, even though we should be summing them.

    SEE ALSO: https://github.com/deepspeedai/DeepSpeed/issues/7107
    (If this feature request gets implemented, we can remove this workaround
    and simply modify the DeepSpeed config to use a SUM gradient all-reduce op.)
    """
    for n, lp in self.model.named_parameters():
        scale_full_hp_grad(lp, self.world_size)


def log_grad_norm_cb(self: BiencoderTrainer) -> None:
    """Post-step callback to log the gradient norm.

    NOTE: Requires ZeRO stage 1 or 2, because we grab the gradient norm from
    `DeepSpeedZeroOptimizer._global_grad_norm`.
    """
    # Only trigger if we're responsible for wandb logging.
    if not self.is_wandb_logger:
        return

    # Gate the import on confirming we're using wandb.
    import wandb

    # Get the grad norm from the DeepSpeed Engine.
    if not isinstance(self.model, DeepSpeedEngine):
        raise ValueError("log_grad_norm_cb requires a DeepSpeedEngine model.")
    deepspeed_engine: DeepSpeedEngine = self.model
    grad_norm = deepspeed_engine.get_global_grad_norm()
    if grad_norm is None:
        return
    if isinstance(grad_norm, Tensor):
        grad_norm = grad_norm.item()

    # Log.
    wandb.log({"train/gradient_norm": grad_norm}, step=self.global_step)


def eval_and_log_cb(self: BiencoderTrainer) -> None:
    """Post-step callback to evaluate and log the model."""
    if self.config.eval_frequency == 0 or self.global_step % self.config.eval_frequency != 0:
        return
    assert self.eval_dataloader_map is not None, "Missing eval loaders"
    if len(self.eval_dataloader_map) == 0:
        # Skip eval if we have nothing to eval on.
        return
    metrics = {}
    initial_train_mode = self.model.training
    try:
        self.model.train(mode=False)
        for eval_name, eval_loader in self.eval_dataloader_map.items():
            em_list = []
            for eval_batch in tqdm(eval_loader, desc=f"eval/{eval_name}", unit="batch"):
                eval_metrics = self.eval(eval_batch)
                em_list.append(eval_metrics)
            avg_metrics = {f"eval/{eval_name}/{k}": sum(em[k] for em in em_list) / len(em_list) for k in eval_metrics}
            metrics.update(avg_metrics)
    finally:
        self.model.train(mode=initial_train_mode)
    if self.is_wandb_logger:
        import wandb

        wandb.log(metrics, step=self.global_step)
    logger.info(f"Global Step: {self.global_step}/{self.training_horizon} Eval: {metrics}")


def splade_flops_warmup_cb(self: BiencoderTrainer) -> None:
    """Pre-step callback to linearly warm up SPLADE FLOPs weights."""
    if not hasattr(self.config, "_splade_flops_warmup_steps"):
        # Initialize warmup config on first call
        self.config._splade_flops_warmup_steps = int(os.environ.get("SPLADE_FLOPS_WARMUP_STEPS", "2000"))
        self.config._splade_flops_weight_query_target = self.config.splade_flops_weight_query
        self.config._splade_flops_weight_doc_target = self.config.splade_flops_weight_doc

    if self.global_step < self.config._splade_flops_warmup_steps:
        # Linear warmup from 0 to target
        warmup_factor = self.global_step / self.config._splade_flops_warmup_steps
        self.config.splade_flops_weight_query = warmup_factor * self.config._splade_flops_weight_query_target
        self.config.splade_flops_weight_doc = warmup_factor * self.config._splade_flops_weight_doc_target
    else:
        # Restore target weights after warmup
        self.config.splade_flops_weight_query = self.config._splade_flops_weight_query_target
        self.config.splade_flops_weight_doc = self.config._splade_flops_weight_doc_target


class BiencoderTrainer(Trainer):
    name = "biencoder"
    config: BiencoderTrainerConfig
    data_factory: ContrastivePretokenizedDataFactory
    model_factory: BiencoderModelFactory
    checkpoint_engine: BiencoderS3CheckpointEngine
    optimizer_factory: FusedAdamOptimizerFactory
    scheduler_factory: Union[WSDSchedulerFactory, HFSchedulerFactory]
    tokenizer_factory: FakeTokenizerFactory
    count_total_queries_seen: int = 0
    count_total_documents_seen: int = 0
    callbacks: List[Tuple[str, Callable]] = [
        ("pre-step", splade_flops_warmup_cb),
        ("post-backward", rescale_grad_cb),
        ("post-step", log_grad_norm_cb),
        ("post-step", eval_and_log_cb),
    ]

    @property
    def is_wandb_logger(self) -> bool:
        return self.global_rank == 0 and self.config.wandb.enable

    def _recreate_dataloader_for_resume(self, start_batch_idx: int) -> None:
        """Recreate dataloader to skip batches when resuming."""
        if start_batch_idx > 0:
            logger.info(f"Recreating dataloader to skip {start_batch_idx} batches for resume")
            # Create new data factory with start_batch_idx
            data_factory = self.config.data.factory(self)
            self.train_dataloader, _ = data_factory(start_batch_idx=start_batch_idx)  # type: ignore[call-arg]

    def pre_train_callback(self) -> None:
        # Synchronize all ranks before starting training
        # This is especially important when resuming from checkpoint
        if self.world_size > 1:
            logger.info(f"Rank {self.global_rank} synchronizing before training starts...")
            torch.distributed.barrier()
            logger.info(f"Rank {self.global_rank} synchronized, starting training")

        # Log training configuration for debugging
        if self.global_rank == 0:
            logger.info(
                f"Starting training from batch_idx={self.train_batch_idx}, "
                f"log_interval={self.config.train_log_iter_interval}"
            )

        # Turn on weights and biases on the master worker.
        if self.is_wandb_logger:
            import wandb

            wandb.init(
                project=self.config.wandb.project,
                config=self.config.model_dump(),
                name=self.config.wandb.name,
                dir="/tmp/wandb",
                save_code=False,
            )

    def forward_and_gather(self, batch: ContrastiveLearningBatch) -> Tuple[Tensor, Tensor, Tensor]:
        # `self.model` is a `DeepSpeedEngine` that wraps a `Biencoder`.
        #   for type-checking, casting to `Biencoder` is helpful, but
        #   runtime checks like `isinstance` will fail.
        self.model = cast(Biencoder, self.model)
        assert isinstance(self.config, BiencoderTrainerConfig), f"{type(self.config)=}"

        # Move the batch to the GPU and densify the relevance labels matrix.
        batch = batch.to_device(self.device)
        relations = batch.relevance_labels.to_dense()

        # Run the model forward pass.
        query_embeddings, document_embeddings = self.model(
            query_input_ids=batch.query_tokens,
            query_attention_mask=batch.query_attention_mask,
            document_input_ids=batch.document_tokens,
            document_attention_mask=batch.document_attention_mask,
        )

        # Gather embeddings across all devices.
        query_embeddings = gather_embeddings(
            embeddings=query_embeddings,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )
        document_embeddings = gather_embeddings(
            embeddings=document_embeddings,
            global_rank=self.global_rank,
            world_size=self.world_size,
        )

        return query_embeddings, document_embeddings, relations

    @torch.no_grad()
    def eval(self, batch: ContrastiveLearningBatch) -> Dict[str, float]:
        query_embeddings, document_embeddings, relations = self.forward_and_gather(batch)
        # Access underlying Biencoder if wrapped by DeepSpeed
        model_core = getattr(self.model, "module", self.model)
        if self.config.use_in_batch_negatives:
            relations[relations == 0] = -1
        if isinstance(model_core, Biencoder) and model_core.pooling == "splade":
            scores = torch.matmul(query_embeddings, document_embeddings.transpose(0, 1))
        else:
            q_emb = F.normalize(query_embeddings, dim=1)
            d_emb = F.normalize(document_embeddings, dim=1)
            scores = torch.matmul(q_emb, d_emb.transpose(0, 1))
        loss_infonce = info_nce_loss(scores, relations=relations, temperature=self.config.loss_temperature).item()
        metrics = {"infoNCE": loss_infonce}
        if isinstance(model_core, Biencoder) and model_core.pooling == "splade":
            thr = self.config.splade_nnz_threshold
            avg_terms_q = (query_embeddings > thr).to(torch.float32).sum(dim=1).mean().item()
            avg_terms_d = (document_embeddings > thr).to(torch.float32).sum(dim=1).mean().item()
            metrics.update({"avg_terms_query": avg_terms_q, "avg_terms_doc": avg_terms_d})
        return metrics

    def loss(self, batch: ContrastiveLearningBatch) -> Tensor:
        # Count the number of queries and documents seen in this batch.
        global_batch_size_query = self.world_size * batch.query_tokens.size(0)
        global_batch_size_doc = self.world_size * batch.document_tokens.size(0)
        self.count_total_queries_seen += global_batch_size_query
        self.count_total_documents_seen += global_batch_size_doc

        # Forward pass, gathering embeddings so each GPU has the full picture.
        query_embeddings, document_embeddings, relations = self.forward_and_gather(batch)

        # InfoNCE loss (SPLADE uses raw dot product; dense uses MRL path).
        model_core = getattr(self.model, "module", self.model)
        if self.config.use_in_batch_negatives:
            relations[relations == 0] = -1
        if isinstance(model_core, Biencoder) and model_core.pooling == "splade":
            scores = torch.matmul(query_embeddings, document_embeddings.transpose(0, 1))
            loss = info_nce_loss(scores, relations=relations, temperature=self.config.loss_temperature)
            loss_base = loss
            loss_truncated = None
        else:
            loss, loss_base, loss_truncated = one_size_truncated_mrl_info_nce_loss(
                query_embeddings=query_embeddings,
                document_embeddings=document_embeddings,
                relations=relations,
                truncated_dimension=self.config.mrl_dim,
                temperature=self.config.loss_temperature,
            )

        # Remove L1 regularizer in favor of SPLADE v2 FLOPs regularizers per side.
        # Keep code path off unless legacy weight is set.
        if isinstance(model_core, Biencoder) and model_core.pooling == "splade" and self.config.splade_reg_weight > 0:
            # No-op by default; legacy users can still get old behavior if they set this.
            reg_q = query_embeddings.abs().sum(dim=1).mean()
            reg_d = document_embeddings.abs().sum(dim=1).mean()
            loss_reg = self.config.splade_reg_weight * (reg_q + reg_d)
            loss = loss + loss_reg

        # SPLADE v2 FLOPs regularization with separate query and document terms.
        if isinstance(model_core, Biencoder) and model_core.pooling == "splade":
            if self.config.splade_flops_weight_query > 0:
                flops_q = model_core.compute_splade_flops_query_cached()
                loss = loss + self.config.splade_flops_weight_query * flops_q
            if self.config.splade_flops_weight_doc > 0:
                flops_d = model_core.compute_splade_flops_doc_cached()
                loss = loss + self.config.splade_flops_weight_doc * flops_d

        # Weights and Biases logging.
        # NOTE: We log more than is feasible to do in a callback, so we do it here
        # in the `loss` function.
        if self.is_wandb_logger:
            import wandb

            # NOTE: Total loss, learning rate, and global step are logged separately
            # in a built-in callback.
            metrics = {
                "train/examples_query": self.count_total_queries_seen,
                "train/examples_doc": self.count_total_documents_seen,
                "train/batch_size_query": global_batch_size_query,
                "train/batch_size_doc": global_batch_size_doc,
                "train/loss_no_truncate": loss_base.item(),
            }
            # SPLADE average active terms per query/doc for this step.
            if isinstance(model_core, Biencoder) and model_core.pooling == "splade":
                thr = self.config.splade_nnz_threshold
                metrics["train/avg_terms_query"] = (query_embeddings > thr).to(torch.float32).sum(dim=1).mean().item()
                metrics["train/avg_terms_doc"] = (document_embeddings > thr).to(torch.float32).sum(dim=1).mean().item()
            if loss_truncated is not None:
                truncated_loss_name = f"train/loss_truncate_{self.config.mrl_dim}"
                metrics[truncated_loss_name] = loss_truncated.item()
            if isinstance(model_core, Biencoder) and model_core.pooling == "splade":
                if self.config.splade_reg_weight > 0:
                    metrics["train/splade_l1_loss"] = loss_reg.item()
                    metrics["train/splade_l1_loss_weight"] = self.config.splade_reg_weight
                else:
                    metrics["train/splade_l1_loss"] = 0.0
                    metrics["train/splade_l1_loss_weight"] = 0.0
                if self.config.splade_flops_weight_query > 0:
                    metrics["train/splade_query_flops_loss"] = flops_q.item()
                    metrics["train/splade_query_flops_loss_weight"] = self.config.splade_flops_weight_query
                else:
                    metrics["train/splade_query_flops_loss"] = 0.0
                    metrics["train/splade_query_flops_loss_weight"] = 0.0
                if self.config.splade_flops_weight_doc > 0:
                    metrics["train/splade_doc_flops_loss"] = flops_d.item()
                    metrics["train/splade_doc_flops_loss_weight"] = self.config.splade_flops_weight_doc
                else:
                    metrics["train/splade_doc_flops_loss"] = 0.0
                    metrics["train/splade_doc_flops_loss_weight"] = 0.0
            wandb.log(metrics, step=self.global_step)

        return loss


def gather_embeddings(embeddings: Tensor, global_rank: int, world_size: int) -> Tensor:
    embeddings = embeddings.contiguous()
    if world_size == 1:
        return embeddings
    assert dist.is_initialized()
    tensor_list = [torch.empty_like(embeddings) for _ in range(world_size)]
    dist.all_gather(tensor_list, embeddings)
    tensor_list[global_rank] = embeddings
    out = torch.cat(tensor_list, dim=0)
    return out


def scale_full_hp_grad(param, scale_factor: float) -> None:
    """In-place gradient scaling for DeepSpeed Zero stages 1 and 2."""
    assert hasattr(param, "_hp_mapping"), "Are you not in Zero 1 or 2?"
    hp_mapping = param._hp_mapping
    if hp_mapping is not None:
        lp_grad_fragment = hp_mapping.get_lp_grad_fragment(param._index_in_param_group)
        lp_grad_fragment.data.mul_(scale_factor)
