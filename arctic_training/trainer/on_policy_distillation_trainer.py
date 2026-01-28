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

"""On-Policy Distillation Trainer.

This module implements on-policy distillation training where:
1. The student model generates its own trajectories
2. The teacher model provides per-token supervision via reverse KL divergence
3. The student is updated to minimize the reverse KL against the teacher

This approach combines the on-policy relevance of RL with the dense reward
signal of distillation, enabling efficient training of smaller models.

Reference: https://thinkingmachines.ai/blog/on-policy-distillation/
"""

from typing import Dict
from typing import Tuple
from typing import Union

import deepspeed
import torch
import torch.nn.functional as F

from arctic_training.checkpoint.ds_engine import DSCheckpointEngine
from arctic_training.checkpoint.hf_engine import HFCheckpointEngine
from arctic_training.config.on_policy_distillation import OnPolicyDistillationTrainerConfig
from arctic_training.data.on_policy_distillation_factory import OnPolicyDistillationDataFactory
from arctic_training.logging import logger
from arctic_training.model.hf_factory import HFModelFactory
from arctic_training.model.liger_factory import LigerModelFactory
from arctic_training.optimizer.adam_factory import CPUAdamOptimizerFactory
from arctic_training.optimizer.adam_factory import FusedAdamOptimizerFactory
from arctic_training.scheduler.hf_factory import HFSchedulerFactory
from arctic_training.tokenizer.hf_factory import HFTokenizerFactory
from arctic_training.trainer.trainer import Trainer
from arctic_training.trainer.utils import disable_dropout_in_model
from arctic_training.trainer.utils import to_device


def init_teacher_model(self: "OnPolicyDistillationTrainer") -> None:
    """Initialize the in-memory teacher model for logprob computation.

    This callback is called post-init to load the teacher model and
    wrap it with DeepSpeed for efficient inference.
    """
    config = self.config

    # Create teacher model using the same factory pattern as DPO's ref_model
    teacher_model_factory = config.teacher_model.factory(
        trainer=self, model_config=config.teacher_model
    )
    self.teacher_model = teacher_model_factory()

    # Wrap with DeepSpeed for efficient inference
    self.teacher_model, *_ = deepspeed.initialize(
        model=self.teacher_model,
        config=config.teacher_deepspeed,
    )

    # Disable dropout for stable distillation signal
    if config.disable_teacher_dropout:
        disable_dropout_in_model(self.teacher_model)

    logger.info("Teacher model initialized for on-policy distillation")


class OnPolicyDistillationTrainer(Trainer):
    """Trainer for On-Policy Distillation.

    On-policy distillation trains a student model by having it generate
    trajectories and using a teacher model to provide per-token feedback
    via reverse KL divergence.

    The loss function is:
        L = E[log π_student(x) - log π_teacher(x)]

    where x are tokens sampled from π_student.

    This is "mode-seeking" - the student learns to approximate the teacher's
    behavior specifically in the states the student visits, making it robust
    to compounding errors in generation.

    Attributes:
        teacher_model: In-memory teacher model for computing logprobs
    """

    name = "on_policy_distillation"
    config: OnPolicyDistillationTrainerConfig
    data_factory: OnPolicyDistillationDataFactory
    model_factory: Union[HFModelFactory, LigerModelFactory]
    teacher_model_factory: Union[HFModelFactory, LigerModelFactory]
    checkpoint_engine: Union[DSCheckpointEngine, HFCheckpointEngine]
    optimizer_factory: Union[FusedAdamOptimizerFactory, CPUAdamOptimizerFactory]
    scheduler_factory: HFSchedulerFactory
    tokenizer_factory: HFTokenizerFactory

    teacher_model: torch.nn.Module

    callbacks = [
        ("post-init", init_teacher_model),
    ]

    def generate_trajectories(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate trajectories from the student model.

        Generates `num_rollouts_per_prompt` completions for each prompt to increase
        sample diversity and GPU utilization.

        Args:
            input_ids: Prompt input IDs (batch_size, prompt_len)
            attention_mask: Attention mask for prompts
            prompt_lengths: Original prompt lengths before padding

        Returns:
            Tuple of:
                - generated_ids: Full sequences (batch_size * num_rollouts, seq_len)
                - labels: Token IDs with -100 for prompt/padding positions
                - attention_mask: Attention mask for generated sequence
        """
        num_rollouts = self.config.num_rollouts_per_prompt
        batch_size = input_ids.size(0)

        # Repeat each prompt num_rollouts times for multiple completions per prompt
        # [p1, p2, p3, p4] with num_rollouts=2 -> [p1, p1, p2, p2, p3, p3, p4, p4]
        if num_rollouts > 1:
            input_ids = input_ids.repeat_interleave(num_rollouts, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_rollouts, dim=0)
            prompt_lengths = prompt_lengths.repeat_interleave(num_rollouts, dim=0)

        expanded_batch_size = input_ids.size(0)  # batch_size * num_rollouts

        # Put model in eval mode for generation (no dropout)
        self.model.eval()

        with torch.no_grad():
            # Generate using the model
            generated_ids = self.model_unwrapped.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.generation_temperature,
                top_p=self.config.generation_top_p,
                top_k=self.config.generation_top_k if self.config.generation_top_k > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        # Put model back in train mode
        self.model.train()

        # Create labels tensor with -100 for prompt positions (like TRL)
        # This allows using cross_entropy's ignore_index for masking
        padded_prompt_len = input_ids.size(1)
        gen_seq_len = generated_ids.size(1)

        # Start with all -100 (ignore all)
        labels = torch.full_like(generated_ids, -100)

        # Create attention mask for generated sequence
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        gen_attention_mask = (generated_ids != pad_token_id).long()

        # Fill in completion tokens (after prompt) for each sequence
        for i in range(expanded_batch_size):
            # Completion starts after the padded prompt
            comp_start = padded_prompt_len

            # Find completion end (first padding or EOS after prompt)
            comp_end = gen_seq_len
            for j in range(comp_start, gen_seq_len):
                if generated_ids[i, j] == pad_token_id:
                    comp_end = j
                    break
                if generated_ids[i, j] == self.tokenizer.eos_token_id:
                    comp_end = j + 1  # Include EOS token
                    break

            # Set labels for completion tokens (non -100)
            if comp_end > comp_start:
                labels[i, comp_start:comp_end] = generated_ids[i, comp_start:comp_end]

        return generated_ids, labels, gen_attention_mask

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the on-policy distillation loss.

        This is the main loss function that:
        1. Generates trajectories from the student
        2. Computes student and teacher log probabilities
        3. Computes reverse KL divergence

        Uses label masking (-100) for clean handling of variable-length completions.

        Args:
            batch: Batch containing input_ids, attention_mask, prompt_lengths

        Returns:
            Scalar loss value
        """
        batch = to_device(batch, self.device)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_lengths = batch["prompt_lengths"]

        # Step 1: Generate trajectories from student (returns labels with -100 for prompts)
        generated_ids, labels, gen_attention_mask = self.generate_trajectories(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
        )

        # Check if any completions were generated
        mask = (labels != -100)
        num_completion_tokens = mask.sum()
        if num_completion_tokens == 0:
            logger.warning("No completions generated, returning zero loss")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Step 2: Forward pass through student model
        student_outputs = self.model(
            input_ids=generated_ids,
            attention_mask=gen_attention_mask,
            use_cache=False,
        )
        student_logits = student_outputs.logits

        # Step 3: Forward pass through teacher model (no grad)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=generated_ids,
                attention_mask=gen_attention_mask,
                use_cache=False,
            )
            teacher_logits = teacher_outputs.logits

        # Step 4: Compute logprobs using fused cross_entropy (shifted for next-token prediction)
        # Shift: logits[t] predicts token at position t+1, so we compare logits[:-1] with labels[1:]
        shift_student_logits = student_logits[:, :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten for cross_entropy
        vocab_size = shift_student_logits.size(-1)
        flat_student_logits = shift_student_logits.view(-1, vocab_size)
        flat_teacher_logits = shift_teacher_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)

        # Compute per-token logprobs using fused cross_entropy
        # cross_entropy returns -log_prob, so negate; ignore_index=-100 handles masking
        student_logprobs = -F.cross_entropy(
            flat_student_logits, flat_labels, ignore_index=-100, reduction='none'
        )
        with torch.no_grad():
            teacher_logprobs = -F.cross_entropy(
                flat_teacher_logits, flat_labels, ignore_index=-100, reduction='none'
            )

        # Step 5: Compute on-policy distillation loss using policy gradient
        # Reference: https://thinkingmachines.ai/blog/on-policy-distillation/
        #
        # The key insight: use -reverse_kl as ADVANTAGE in policy gradient, not as direct loss.
        #
        # reverse_kl = student_logprob - teacher_logprob
        # advantage = -reverse_kl = teacher_logprob - student_logprob
        # 
        # Policy gradient loss: loss = -advantage * log_prob
        #   = -(teacher_logprob - student_logprob) * student_logprob
        #
        # This gives correct gradients:
        # - When teacher > student (advantage > 0): increase student_logprob
        # - When teacher < student (advantage < 0): decrease student_logprob
        
        shift_mask = (flat_labels != -100)
        num_tokens = shift_mask.sum()
        
        if num_tokens > 0:
            valid_student_logprobs = student_logprobs[shift_mask]
            valid_teacher_logprobs = teacher_logprobs[shift_mask]
            
            # Advantage = teacher's preference - student's confidence (detached for stable training)
            # Positive advantage: teacher likes this token more than student expects
            # Negative advantage: student is overconfident relative to teacher
            advantage = (valid_teacher_logprobs - valid_student_logprobs.detach())
            
            # Policy gradient loss: maximize student logprob weighted by advantage
            # loss = -E[advantage * log p_student]
            loss = -(advantage * valid_student_logprobs).mean() * self.config.beta
            
            # Track reverse KL for monitoring
            per_token_kl = valid_student_logprobs - valid_teacher_logprobs
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            per_token_kl = torch.zeros(1, device=self.device)

        # Log detailed metrics for monitoring training progress
        with torch.no_grad():
            batch_size = generated_ids.size(0)
            avg_completion_len = num_completion_tokens.float() / batch_size

            if num_tokens > 0:
                # Log probability metrics (key for understanding training dynamics)
                student_logprob_mean = valid_student_logprobs.mean()
                teacher_logprob_mean = valid_teacher_logprobs.mean()
                logprob_gap = student_logprob_mean - teacher_logprob_mean

                # Group metrics by concept using / prefix for W&B panel grouping
                # logprob/ group - raw log probabilities
                self.metrics.record("logprob/student", student_logprob_mean.item())
                self.metrics.record("logprob/teacher", teacher_logprob_mean.item())
                self.metrics.record("logprob/gap", logprob_gap.item())

                # perplexity/ group - easier to interpret
                student_ppl = torch.exp(-student_logprob_mean)
                teacher_ppl = torch.exp(-teacher_logprob_mean)
                self.metrics.record("perplexity/student", student_ppl.item())
                self.metrics.record("perplexity/teacher", teacher_ppl.item())

                # distill/ group - distillation-specific metrics
                mean_kl = per_token_kl.mean()
                prob_ratio = torch.exp(logprob_gap)
                mean_advantage = (valid_teacher_logprobs - valid_student_logprobs).mean()
                self.metrics.record("distill/reverse_kl", mean_kl.item())
                self.metrics.record("distill/prob_ratio", prob_ratio.item())
                self.metrics.record("distill/advantage", mean_advantage.item())

                # generation/ group
                self.metrics.record("generation/avg_length", avg_completion_len.item())
            else:
                self.metrics.record("generation/avg_length", 0.0)

        return loss

    def step(self, batch: Dict[str, torch.Tensor]) -> None:
        """Execute a single training step.

        Overrides the base step to handle the unique requirements of
        on-policy distillation (generation + training).
        """
        self.model.train()

        loss = self.loss(batch)

        self.backward(loss)

        def maybe_item(v):
            return v.item() if torch.is_tensor(v) else v

        self.metrics.record("loss", maybe_item(loss))

        self.model.step()

        self.checkpoint()

        # Update step counters
        self.global_step = self.model.global_steps
        self.global_step_this_run = self.global_step - self.global_step_at_start_this_run

    def evaluate(self) -> None:
        """Evaluation loop with detailed metrics for on-policy distillation.
        
        Note: We intentionally don't use @callback_wrapper here to avoid
        the base class's evaluate() being called instead of this override.

        Computes loss and various metrics on the validation set to track:
        - Whether student is learning to match teacher
        - Generation quality metrics
        - KL divergence trends
        """
        if self.eval_dataloader is None:
            logger.warning("No evaluation dataloader, skipping evaluation")
            return

        self.model.eval()
        self.teacher_model.eval()

        # Accumulators for metrics
        total_loss = 0.0
        total_kl = 0.0
        total_student_logprob = 0.0
        total_teacher_logprob = 0.0
        total_advantage = 0.0
        total_completion_len = 0.0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for eval_batch in self.eval_batches:
                eval_batch = to_device(eval_batch, self.device)

                input_ids = eval_batch["input_ids"]
                attention_mask = eval_batch["attention_mask"]
                prompt_lengths = eval_batch["prompt_lengths"]

                # Generate trajectories
                generated_ids, labels, gen_attention_mask = self.generate_trajectories(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_lengths=prompt_lengths,
                )

                mask = (labels != -100)
                num_completion_tokens = mask.sum()
                if num_completion_tokens == 0:
                    continue

                # Forward passes
                student_logits = self.model(
                    input_ids=generated_ids,
                    attention_mask=gen_attention_mask,
                    use_cache=False,
                ).logits

                teacher_logits = self.teacher_model(
                    input_ids=generated_ids,
                    attention_mask=gen_attention_mask,
                    use_cache=False,
                ).logits

                # Compute logprobs
                shift_student_logits = student_logits[:, :-1, :].contiguous()
                shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                vocab_size = shift_student_logits.size(-1)
                flat_student_logits = shift_student_logits.view(-1, vocab_size)
                flat_teacher_logits = shift_teacher_logits.view(-1, vocab_size)
                flat_labels = shift_labels.view(-1)

                student_logprobs = -F.cross_entropy(
                    flat_student_logits, flat_labels, ignore_index=-100, reduction='none'
                )
                teacher_logprobs = -F.cross_entropy(
                    flat_teacher_logits, flat_labels, ignore_index=-100, reduction='none'
                )

                shift_mask = (flat_labels != -100)
                num_tokens = shift_mask.sum().item()

                if num_tokens > 0:
                    valid_student_logprobs = student_logprobs[shift_mask]
                    valid_teacher_logprobs = teacher_logprobs[shift_mask]
                    
                    # Compute policy gradient loss (same as training)
                    advantage = valid_teacher_logprobs - valid_student_logprobs
                    batch_loss = -(advantage * valid_student_logprobs).mean() * self.config.beta
                    total_loss += batch_loss.item()
                    
                    # Track reverse KL for monitoring
                    per_token_kl = valid_student_logprobs - valid_teacher_logprobs

                    total_kl += per_token_kl.sum().item()
                    total_student_logprob += valid_student_logprobs.sum().item()
                    total_teacher_logprob += valid_teacher_logprobs.sum().item()
                    total_advantage += advantage.sum().item()
                    total_tokens += num_tokens
                    total_completion_len += num_completion_tokens.item()
                    num_batches += 1

                # Clean up large tensors to prevent OOM during eval
                del student_logits, teacher_logits
                del shift_student_logits, shift_teacher_logits
                del flat_student_logits, flat_teacher_logits
                del student_logprobs, teacher_logprobs
                del generated_ids, labels, gen_attention_mask
                torch.cuda.empty_cache()

        # Log aggregated metrics
        if num_batches > 0 and total_tokens > 0:
            avg_loss = total_loss / num_batches
            avg_kl = total_kl / total_tokens
            avg_student_logprob = total_student_logprob / total_tokens
            avg_teacher_logprob = total_teacher_logprob / total_tokens
            avg_advantage = total_advantage / total_tokens
            avg_completion_len = total_completion_len / (num_batches * self.config.micro_batch_size * self.config.num_rollouts_per_prompt)

            # Perplexity
            student_ppl = torch.exp(torch.tensor(-avg_student_logprob)).item()
            teacher_ppl = torch.exp(torch.tensor(-avg_teacher_logprob)).item()
            prob_ratio = torch.exp(torch.tensor(avg_student_logprob - avg_teacher_logprob)).item()

            # Record all eval metrics with eval/ prefix (W&B groups by first segment)
            self.metrics.record("loss/eval", avg_loss)
            self.metrics.record("eval/logprob_student", avg_student_logprob)
            self.metrics.record("eval/logprob_teacher", avg_teacher_logprob)
            self.metrics.record("eval/logprob_gap", avg_student_logprob - avg_teacher_logprob)
            self.metrics.record("eval/perplexity_student", student_ppl)
            self.metrics.record("eval/perplexity_teacher", teacher_ppl)
            self.metrics.record("eval/reverse_kl", avg_kl)
            self.metrics.record("eval/prob_ratio", prob_ratio)
            self.metrics.record("eval/advantage", avg_advantage)
            self.metrics.record("eval/avg_completion_length", avg_completion_len)

            logger.info(
                f"Eval | loss: {avg_loss:.4f} | kl: {avg_kl:.4f} | adv: {avg_advantage:.4f} | "
                f"student_ppl: {student_ppl:.2f} | teacher_ppl: {teacher_ppl:.2f} | "
                f"prob_ratio: {prob_ratio:.4f} | comp_len: {avg_completion_len:.1f}"
            )
