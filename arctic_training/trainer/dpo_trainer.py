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
from typing import Union

import torch

from arctic_training.checkpoint.ds_engine import DSCheckpointEngine
from arctic_training.checkpoint.hf_engine import HFCheckpointEngine
from arctic_training.data.dpo_factory import DPODataFactory
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.model.liger_factory import LigerModelFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.registry import register
from arctic_training.scheduler.hf_factory import HFSchedulerFactory
from arctic_training.tokenizer.hf_factory import HFTokenizerFactory
from arctic_training.trainer import Trainer



def to_device(batch: Dict, device: str) -> Dict:
    output = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            output[k] = v.to(device)
    return output



def get_logprobs(
        logits: torch.Tensor, labels: torch.Tensor,
        ignore_label_index: int
    ) -> Tuple[float, float]:
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
            f"Logits (batch and sequence length dim) {logits.shape[:-1]} and labels must have the same shape {labels.shape}."
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

class DPOTrainerConfig:
    ref_model: ModelConfig
    beta: float
    ignore_label_index: int = -100
    label_smoothing: float = 0.0
    """ Model configuration. """


class DPOTrainer(Trainer):
    name = "dpo"
    config: DPOTrainerConfig
    model_factory: Union[HFModelFactory, LigerModelFactory]
    ref_model_factory: Union[HFModelFactory, LigerModelFactory]
    checkpoint_engine: Union[DSCheckpointEngine, HFCheckpointEngine]
    optimizer_factory: FusedAdamOptimizerFactory
    scheduler_factory: HFSchedulerFactory
    tokenizer_factory: HFTokenizerFactory
    callbacks = [("post-init", init_ref_model)]


    def init_ref_model(self):
        ref_model_factory = self.config.ref_model.factory(trainer=self, config=self.config.ref_model) # Be explicit about which model config to use
        self.ref_model = ref_model_factory()

    def forward_model(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logits = outputs.logits
        logprobs, completion_sizes = get_logprobs(logits, batch["labels"],
            self.config.ignore_label_index)
        return logits, logprobs, completion_sizes

    def forword_reference_model(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            output = self.reference_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            logits = output.logits
            logprobs, completion_sizes = get_logprobs(logits, batch["labels"],
            self.config.ignore_label_index)
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
            -F.logsigmoid(self.config.beta * logits)
            * (1 - self.config.label_smoothing)
            - F.logsigmoid(-self.config.beta * logits) * self.config.label_smoothing
        )

        chosen_rewards = (
            self.config.beta * (chosen_logprobs - ref_chosen_logprobs).detach()
        )
        rejected_rewards = (
            self.config.beta * (rejected_logprobs - ref_rejected_logprobs).detach()
        )

        return losses, chosen_rewards, rejected_rewards


    def loss(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)
        ref_logits, ref_logprobs, _ = self.forword_reference_model(batch)
        logits, logprobs, _ = self.forward_model(batch)
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(logprobs, ref_logprobs)
        reward_acc = (chosen_rewards > rejected_rewards).float().mean()
        # chosen_rewards.mean(), rejected_rewards.mean(), reward_acc
        return losses.mean()
