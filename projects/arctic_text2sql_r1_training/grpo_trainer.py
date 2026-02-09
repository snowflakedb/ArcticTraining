#!/usr/bin/env python3
"""
GRPO Trainer for Text-to-SQL
Implements Group Relative Policy Optimization as described in Arctic-Text2SQL-R1 paper
Paper: https://arxiv.org/abs/2505.20315
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import sqlite3
from dataclasses import dataclass

from arctic_training.trainers.sft_trainer import SFTTrainer, SFTTrainerConfig
from arctic_training.registry import RegistryMeta


@dataclass
class GRPOTrainerConfig(SFTTrainerConfig):
    """Configuration for GRPO trainer"""

    # GRPO-specific hyperparameters from paper
    num_samples_per_prompt: int = 16  # Paper uses 16 rollouts per sample
    temperature: float = 0.8  # Paper uses 0.8 for generation
    kl_coef: float = 0.001  # KL penalty coefficient (β)
    clip_range: float = 0.2  # PPO clipping ratio (ε)

    # Reward computation
    reward_correct: float = 1.0  # Paper: exact match
    reward_executable: float = 0.1  # Paper: executable but wrong
    reward_failed: float = 0.0  # Paper: syntax error

    # Generation parameters
    max_new_tokens: int = 150
    top_p: float = 0.9


class GRPOTrainer(SFTTrainer, metaclass=RegistryMeta, type_tag="grpo"):
    """
    Group Relative Policy Optimization Trainer for Text-to-SQL

    Extends SFTTrainer with RL capabilities following Arctic-Text2SQL-R1 paper.

    Key features:
    - Generates N SQL candidates per prompt
    - Computes execution-based rewards
    - Normalizes advantages within groups (GRPO's key innovation)
    - Updates policy with PPO-style clipping
    """

    def __init__(self, config: GRPOTrainerConfig):
        super().__init__(config)
        self.config = config

        # Store reference model for KL computation
        self.reference_model = None

    def setup(self):
        """Initialize trainer including reference model for KL divergence"""
        super().setup()

        # Clone reference model for KL penalty
        # Reference model stays frozen during training
        if self.config.kl_coef > 0:
            from copy import deepcopy
            self.reference_model = deepcopy(self.model)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

    def generate_candidates(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_samples: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate N SQL candidates per prompt using sampling

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            num_samples: Number of candidates to generate per input

        Returns:
            generated_ids: [batch_size * num_samples, seq_len]
            generated_mask: [batch_size * num_samples, seq_len]
        """
        batch_size = input_ids.shape[0]

        # Repeat inputs for multiple samples
        input_ids_repeated = input_ids.repeat_interleave(num_samples, dim=0)
        attention_mask_repeated = attention_mask.repeat_interleave(num_samples, dim=0)

        # Generate with sampling for diversity
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids_repeated,
                attention_mask=attention_mask_repeated,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Create attention mask for generated sequences
        generated_mask = (outputs != self.tokenizer.pad_token_id).long()

        return outputs, generated_mask

    def compute_rewards(
        self,
        generated_texts: List[str],
        gold_sql: str,
        database_path: str
    ) -> torch.Tensor:
        """
        Compute execution-based rewards following paper's approach

        Reward function (from paper):
        - 1.0: SQL executes correctly and matches gold result (exact match)
        - 0.1: SQL is executable but produces wrong result
        - 0.0: SQL has syntax errors or fails to execute

        Args:
            generated_texts: List of generated SQL queries
            gold_sql: Ground truth SQL query
            database_path: Path to SQLite database

        Returns:
            rewards: Tensor of rewards [num_samples]
        """
        rewards = []

        # Get gold result for comparison
        gold_success, gold_result = self._execute_sql(gold_sql, database_path)

        for sql_text in generated_texts:
            # Extract SQL from generated text
            sql = self._extract_sql(sql_text)

            # Execute and compare
            success, result = self._execute_sql(sql, database_path)

            if success and gold_success and self._compare_results(result, gold_result):
                # Exact match - perfect!
                reward = self.config.reward_correct
            elif success:
                # Executable but wrong result
                reward = self.config.reward_executable
            else:
                # Syntax error or execution failure
                reward = self.config.reward_failed

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _execute_sql(self, sql: str, db_path: str) -> Tuple[bool, Optional[str]]:
        """
        Execute SQL query and return results

        Returns:
            (success, result): success is True if query executed, result is string representation
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return True, str(sorted(result))  # Sort for consistent comparison
        except Exception as e:
            return False, None

    def _extract_sql(self, text: str) -> str:
        """Extract SQL query from model output"""
        import re

        # Look for SQL in code blocks
        sql_match = re.search(r'```sql\n(.*?)\n```', text, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()

        # Look for SELECT/INSERT/UPDATE/DELETE statements
        sql_patterns = [
            r'(SELECT\s+.*?)(?:;|$)',
            r'(INSERT\s+.*?)(?:;|$)',
            r'(UPDATE\s+.*?)(?:;|$)',
            r'(DELETE\s+.*?)(?:;|$)',
        ]

        for pattern in sql_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()

        # Fallback: return as-is
        return text.strip()

    def _compare_results(self, result1: str, result2: str) -> bool:
        """Compare SQL execution results"""
        return result1 == result2

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        batch_size: int,
        num_samples: int
    ) -> torch.Tensor:
        """
        Compute group-relative advantages (GRPO's key innovation)

        Instead of global baseline, normalize within each group of candidates
        generated from the same prompt. This reduces variance.

        Args:
            rewards: [batch_size * num_samples]
            batch_size: Number of prompts
            num_samples: Number of candidates per prompt

        Returns:
            advantages: [batch_size * num_samples]
        """
        # Reshape to [batch_size, num_samples]
        rewards_grouped = rewards.view(batch_size, num_samples)

        # Compute group-relative advantages
        mean = rewards_grouped.mean(dim=1, keepdim=True)  # [batch_size, 1]
        std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8  # [batch_size, 1]

        # Normalize within each group
        advantages = (rewards_grouped - mean) / std

        # Flatten back
        return advantages.view(-1)

    def compute_kl_penalty(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty: KL(π_θ || π_ref)

        Args:
            logprobs: Log probabilities from current policy
            ref_logprobs: Log probabilities from reference policy
            mask: Attention mask [batch_size, seq_len]

        Returns:
            kl: Mean KL divergence
        """
        kl_div = logprobs - ref_logprobs
        kl_masked = kl_div * mask
        kl_mean = kl_masked.sum() / mask.sum()
        return kl_mean

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute GRPO loss

        GRPO objective (from paper):
        J_GRPO(θ) = E[1/N ∑ᵢ min(rᵢAᵢ, clip(rᵢ, 1-ε, 1+ε)Aᵢ)] - β·KL(π_θ||π_ref)

        where:
        - rᵢ = π_θ(aᵢ|sᵢ) / π_old(aᵢ|sᵢ) is probability ratio
        - Aᵢ is group-relative advantage
        - ε is clip range (0.2 in paper)
        - β is KL coefficient (0.001 in paper)

        Args:
            batch: Dictionary containing:
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - gold_sql: List of gold SQL queries
                - database_path: List of database paths

        Returns:
            loss: Scalar loss tensor
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = input_ids.shape[0]
        num_samples = self.config.num_samples_per_prompt

        # Step 1: Generate N candidates per prompt
        generated_ids, generated_mask = self.generate_candidates(
            input_ids, attention_mask, num_samples
        )

        # Step 2: Decode and compute rewards
        generated_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        rewards_list = []
        for i in range(batch_size):
            start_idx = i * num_samples
            end_idx = (i + 1) * num_samples
            batch_texts = generated_texts[start_idx:end_idx]

            rewards = self.compute_rewards(
                batch_texts,
                batch['gold_sql'][i],
                batch['database_path'][i]
            )
            rewards_list.append(rewards)

        rewards = torch.cat(rewards_list, dim=0).to(self.device)

        # Step 3: Compute group-relative advantages
        advantages = self.compute_advantages(rewards, batch_size, num_samples)

        # Step 4: Compute log probabilities from current policy
        outputs = self.model(
            input_ids=generated_ids,
            attention_mask=generated_mask,
        )
        logits = outputs.logits
        logprobs = F.log_softmax(logits, dim=-1)

        # Get log probs of generated tokens
        # Shift to align predictions with targets
        shift_logprobs = logprobs[:, :-1, :].contiguous()
        shift_labels = generated_ids[:, 1:].contiguous()

        # Gather log probs of actual tokens
        token_logprobs = torch.gather(
            shift_logprobs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask and sum
        shift_mask = generated_mask[:, 1:].contiguous()
        sequence_logprobs = (token_logprobs * shift_mask).sum(dim=1)

        # Step 5: Compute log probs from reference policy
        if self.reference_model is not None:
            with torch.no_grad():
                ref_outputs = self.reference_model(
                    input_ids=generated_ids,
                    attention_mask=generated_mask,
                )
                ref_logits = ref_outputs.logits
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)

                shift_ref_logprobs = ref_logprobs[:, :-1, :].contiguous()
                ref_token_logprobs = torch.gather(
                    shift_ref_logprobs, 2, shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                ref_sequence_logprobs = (ref_token_logprobs * shift_mask).sum(dim=1)
        else:
            # No reference model, use current policy as reference
            ref_sequence_logprobs = sequence_logprobs.detach()

        # Step 6: Compute probability ratios
        ratio = torch.exp(sequence_logprobs - ref_sequence_logprobs)

        # Step 7: Clipped surrogate objective (PPO)
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_range,
            1.0 + self.config.clip_range
        ) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # Step 8: KL penalty
        if self.config.kl_coef > 0 and self.reference_model is not None:
            kl_penalty = self.compute_kl_penalty(
                token_logprobs, ref_token_logprobs, shift_mask
            )
            total_loss = policy_loss + self.config.kl_coef * kl_penalty
        else:
            total_loss = policy_loss

        # Logging
        self.log_dict({
            'loss/policy': policy_loss.item(),
            'loss/total': total_loss.item(),
            'rewards/mean': rewards.mean().item(),
            'rewards/max': rewards.max().item(),
            'rewards/min': rewards.min().item(),
            'advantages/mean': advantages.mean().item(),
            'advantages/std': advantages.std().item(),
        })

        return total_loss
