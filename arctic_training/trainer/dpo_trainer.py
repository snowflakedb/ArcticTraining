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
from typing import Tuple
from typing import Union
from typing import cast

import deepspeed
import torch

# import torch.distributed as dist
import torch.nn.functional as F
from pydantic import ValidationInfo
from pydantic import field_validator

from arctic_training.callback.logging import post_loss_log_cb
from arctic_training.callback.wandb import init_wandb_project_cb
from arctic_training.callback.wandb import log_wandb_loss_cb
from arctic_training.callback.wandb import teardown_wandb_cb
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


def to_device(batch: Dict, device: str) -> Dict:
    output = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            output[k] = v.to(device)
    return output


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

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)

    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)


def get_eval_ds_config(stage: int = 0) -> Dict[str, Any]:

    data_type = "bfloat16"
    dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "memory_efficient_linear": False,
    }
    return {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 4,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


class DPOTrainerConfig(TrainerConfig):
    ref_model: ModelConfig
    beta: float
    ignore_label_index: int = -100
    label_smoothing: float = 0.0
    reference_model_deepspeed: Dict = {}
    """ Model configuration. """

    @field_validator("ref_model", mode="before")
    @classmethod
    def init_ref_model_config(
        cls, v: Union[Dict, ModelConfig], info: ValidationInfo
    ) -> ModelConfig:
        subconfig = cls._get_subconfig_object(
            v=v,
            info=info,
            get_class_fn=get_registered_model_factory,
            attr_name="ref_model_factory",
        )
        return cast(ModelConfig, subconfig)


def init_ref_model(self: "DPOTrainer") -> None:
    ref_model_factory = self.config.ref_model.factory(
        trainer=self, model_config=self.config.ref_model
    )  # Be explicit about which model config to use
    if self.config.deepspeed["zero_optimization"]["stage"] == 3:
        ds_stage = 3
    else:
        ds_stage = 0
    self.config.reference_model_deepspeed = get_eval_ds_config(stage=ds_stage)

    self.config.reference_model_deepspeed["train_micro_batch_size_per_gpu"] = (
        self.config.deepspeed["train_micro_batch_size_per_gpu"]
    )
    self.config.reference_model_deepspeed["train_batch_size"] = self.config.deepspeed[
        "train_batch_size"
    ]

    self.ref_model = ref_model_factory()
    # wrap the model with deepspeed
    self.ref_model, *_ = deepspeed.initialize(
        model=self.ref_model, config=self.config.reference_model_deepspeed
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
    callbacks = [
        ("post-init", init_ref_model),
        post_loss_log_cb,
        init_wandb_project_cb,
        log_wandb_loss_cb,
        teardown_wandb_cb,
    ]
    ref_model: torch.nn.Module

    def forward_model(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logits = outputs.logits
        logprobs, completion_sizes = get_logprobs(
            logits, batch["labels"], self.config.ignore_label_index
        )
        return logits, logprobs, completion_sizes

    def forward_reference_model(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output = self.ref_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            logits = output.logits
            logprobs, completion_sizes = get_logprobs(
                logits, batch["labels"], self.config.ignore_label_index
            )
        return logits.detach(), logprobs.detach(), completion_sizes.detach()

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

        # tmp_loss = -F.logsigmoid(self.config.beta * logits)
        # neg_tmp_loss = -F.logsigmoid(-self.config.beta * logits)

        chosen_rewards = (
            self.config.beta * (chosen_logprobs - ref_chosen_logprobs).detach()
        )
        rejected_rewards = (
            self.config.beta * (rejected_logprobs - ref_rejected_logprobs).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    def loss(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)
        ref_logits, ref_logprobs, _ = self.forward_reference_model(batch)
        logits, logprobs, _ = self.forward_model(batch)
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(logprobs, ref_logprobs)

        # This is being dropped. Problem?
        # reward_acc = (chosen_rewards > rejected_rewards).float().mean()
        # chosen_rewards.mean(), rejected_rewards.mean(), reward_acc
        return losses.mean()
