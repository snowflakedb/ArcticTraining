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

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import deepspeed
import torch
import torch.nn.functional as F
from pydantic import ValidationInfo
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

try:
    from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss
except (ImportError, ModuleNotFoundError):
    LigerFusedLinearDPOLoss = None

from arctic_training.checkpoint.ds_engine import DSCheckpointEngine
from arctic_training.checkpoint.hf_engine import HFCheckpointEngine
from arctic_training.config.model import ModelConfig
from arctic_training.config.trainer import TrainerConfig
from arctic_training.data.dpo_factory import DPODataFactory
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.model.liger_factory import LigerModelFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.registry import get_registered_model_factory
from arctic_training.scheduler.hf_factory import HFSchedulerFactory
from arctic_training.tokenizer.hf_factory import HFTokenizerFactory
from arctic_training.trainer.trainer import Trainer
from arctic_training.trainer.utils import to_device


def get_logprobs(
    logits: torch.Tensor, labels: torch.Tensor, ignore_label_index: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A Tuple of two tensor of shape ((batch_size,), (batch_size,)) containing the sum of log probabilities of the given labels under the given logits in the first tensor and the number of non-masked tokens in the second tensor.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"Logits (batch and sequence length dim) {logits.shape[:-1]} and labels"
            f" must have the same shape {labels.shape}."
        )

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != ignore_label_index

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == ignore_label_index] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)


class DPOTrainerConfig(TrainerConfig):
    ref_model: ModelConfig
    """
    Defines the reference model used in Direct Preference Optimization (DPO) training.
    The reference model represents the initial, unoptimized language model
    before preference-based fine-tuning. It provides baseline log-likelihoods for
    comparison against the policy model during training.
    """

    beta: float = 0.1
    """
    Parameter controlling the deviation from the reference model.
    Higher beta means less deviation from the reference model.
    """

    ignore_label_index: int = -100
    """ label value for ignored labels. """

    label_smoothing: float = 0.0
    """
    Robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report
    and [Robust DPO](https://huggingface.co/papers/2403.00409) paper that should be between 0.0 and 0.5.
    """

    reference_model_deepspeed: Dict = {}
    """ Model configuration. """

    @field_validator("ref_model", mode="before")
    @classmethod
    def init_ref_model_config(cls, v: Union[Dict, ModelConfig], info: ValidationInfo) -> ModelConfig:
        subconfig = cls._get_subconfig_object(
            v=v,
            info=info,
            get_class_fn=get_registered_model_factory,
            attr_name="ref_model_factory",
        )
        return cast(ModelConfig, subconfig)

    @model_validator(mode="after")
    def update_deepspeed_dpo_config(self) -> Self:
        """Updates deepspeed config for DPO Trainer."""
        self.deepspeed["train_micro_batch_size_per_gpu"] = int(self.micro_batch_size * 2)
        self.deepspeed["train_batch_size"] = int(
            self.micro_batch_size * self.gradient_accumulation_steps * self.world_size * 2
        )
        return self

    @model_validator(mode="after")
    def build_ref_model_deepspeed_config(self) -> Self:
        """Build deepspeed config for reference model."""
        if len(self.reference_model_deepspeed) != 0:
            raise ValueError(
                "Reference model deepspeed config is computed based on the main model"
                " deepspeed config and should not be passed by the user."
            )

        ref_model_deepspeed = dict(
            train_batch_size=self.deepspeed["train_batch_size"],
            train_micro_batch_size_per_gp=self.deepspeed["train_micro_batch_size_per_gpu"],
            steps_per_print=self.deepspeed["steps_per_print"],
            zero_optimization=dict(
                stage=3 if self.deepspeed["zero_optimization"]["stage"] == 3 else 0,
                stage3_param_persistence_threshold=1e4,
                memory_efficient_linear=False,
            ),
            bfloat16=dict(enabled=True),
            gradient_clipping=1.0,
            prescale_gradients=False,
            wall_clock_breakdown=False,
        )
        self.reference_model_deepspeed = ref_model_deepspeed
        return self


def init_ref_model(self: "DPOTrainer") -> None:
    ref_model_factory = self.config.ref_model.factory(
        trainer=self, model_config=self.config.ref_model
    )  # Be explicit about which model config to use
    self.ref_model = ref_model_factory()
    # wrap the model with deepspeed
    self.ref_model, *_ = deepspeed.initialize(model=self.ref_model, config=self.config.reference_model_deepspeed)


def init_liger_dpo_loss(self: "DPOTrainer") -> None:
    if LigerFusedLinearDPOLoss is not None:
        self.liger_dpo_loss = LigerFusedLinearDPOLoss(
            ignore_index=self.config.ignore_label_index, beta=self.config.beta
        )


class DPOTrainer(Trainer):
    name = "dpo"
    config: DPOTrainerConfig
    data_factory: DPODataFactory
    model_factory: Union[HFModelFactory, LigerModelFactory]
    ref_model_factory: Union[HFModelFactory, LigerModelFactory]
    checkpoint_engine: Union[DSCheckpointEngine, HFCheckpointEngine]
    optimizer_factory: FusedAdamOptimizerFactory
    scheduler_factory: HFSchedulerFactory
    tokenizer_factory: HFTokenizerFactory
    ref_model: torch.nn.Module
    liger_dpo_loss: Optional[LigerFusedLinearDPOLoss] = None

    callbacks = [
        ("post-init", init_ref_model),
        ("post-init", init_liger_dpo_loss),
    ]

    def forward_model(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
        )
        logits = outputs.logits.to(torch.float32)
        logprobs, completion_sizes = get_logprobs(logits, batch["labels"], self.config.ignore_label_index)
        return logits, logprobs, completion_sizes, outputs.hidden_states[-1]

    def forward_reference_model(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output = self.ref_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                output_hidden_states=True,
            )
            logits = output.logits.to(torch.float32)
            logprobs, completion_sizes = get_logprobs(logits, batch["labels"], self.config.ignore_label_index)
        return (
            logits.detach(),
            logprobs.detach(),
            completion_sizes.detach(),
            output.hidden_states[-1].detach(),
        )

    def dpo_loss(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DPO Loss: -E_{(x, y_w, y_l)~D}[log preference]
        preference: sigmoid(chosen_reward
            - beta * log(pi_{\theta}(y_l | x) / pi_{ref}(y_l | x)))
        chosen_reward: beta * log(pi_{\theta}(y_w | x) / pi_{ref}(y_w | x))
        rejected_reward:
        """

        batch_size = logprobs.size(0) // 2
        chosen_logprobs = logprobs[:batch_size]
        rejected_logprobs = logprobs[batch_size:]
        ref_chosen_logprobs = ref_logprobs[:batch_size]
        ref_rejected_logprobs = ref_logprobs[batch_size:]

        pi_logratios = chosen_logprobs - rejected_logprobs
        ref_logratios = ref_chosen_logprobs - ref_rejected_logprobs

        logits = pi_logratios - ref_logratios
        losses = (
            -F.logsigmoid(self.config.beta * logits) * (1 - self.config.label_smoothing)
            - F.logsigmoid(-self.config.beta * logits) * self.config.label_smoothing
        )
        chosen_rewards = self.config.beta * (chosen_logprobs - ref_chosen_logprobs).detach()
        rejected_rewards = self.config.beta * (rejected_logprobs - ref_rejected_logprobs).detach()

        return losses, chosen_rewards, rejected_rewards

    def loss(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)

        ref_logits, ref_logprobs, _, ref_hidden_state = self.forward_reference_model(batch)
        logits, logprobs, _, hidden_state = self.forward_model(batch)

        # Activate if we have liger kernel
        if self.liger_dpo_loss is not None:
            losses, _, _ = self.liger_dpo_loss(
                hidden_state,
                self.model.module.lm_head.weight,
                batch["labels"][:, 1:],
                ref_input=ref_hidden_state.detach(),
                ref_weight=self.ref_model.module.lm_head.weight,
            )
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(logprobs, ref_logprobs)

        return losses.mean()
