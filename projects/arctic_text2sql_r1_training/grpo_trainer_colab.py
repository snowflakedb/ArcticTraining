#!/usr/bin/env python3
"""
GRPO Trainer for Text-to-SQL - Standalone Colab Version
Implements Group Relative Policy Optimization as described in Arctic-Text2SQL-R1 paper
Paper: https://arxiv.org/abs/2505.20315

This is a standalone version that works in Google Colab without requiring
the full ArcticTraining framework.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import sqlite3
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class GRPOTrainerConfig:
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

    # Training parameters
    epochs: int = 3
    learning_rate: float = 1e-6
    logging_steps: int = 5
    output_dir: str = "./grpo_trained_model"


class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer for Text-to-SQL
    Standalone version for Google Colab

    Key features:
    - Generates N SQL candidates per prompt
    - Computes execution-based rewards
    - Normalizes advantages within groups (GRPO's key innovation)
    - Updates policy with PPO-style clipping
    """

    def __init__(self, model, tokenizer, config: GRPOTrainerConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

        # Store reference model for KL computation
        self.reference_model = None
        self._setup_reference_model()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Training metrics
        self.training_stats = {
            'losses': [],
            'rewards': [],
            'advantages': []
        }

    def _setup_reference_model(self):
        """Initialize reference model for KL divergence"""
        if self.config.kl_coef > 0:
            from copy import deepcopy
            self.reference_model = deepcopy(self.model)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
            print("✅ Reference model created for KL penalty")

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
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Create attention mask for generated sequences
        generated_mask = (outputs != (self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)).long()

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
                reward = self.config.reward_correct
            elif success:
                reward = self.config.reward_executable
            else:
                reward = self.config.reward_failed

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _execute_sql(self, sql: str, db_path: str) -> Tuple[bool, Optional[str]]:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return True, str(sorted(result))
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
        """
        # Reshape to [batch_size, num_samples]
        rewards_grouped = rewards.view(batch_size, num_samples)

        # Compute group-relative advantages
        mean = rewards_grouped.mean(dim=1, keepdim=True)
        std = rewards_grouped.std(dim=1, keepdim=True) + 1e-8

        # Normalize within each group
        advantages = (rewards_grouped - mean) / std

        # Flatten back
        return advantages.view(-1)

    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute GRPO loss

        GRPO objective (from paper):
        J_GRPO(θ) = E[1/N ∑ᵢ min(rᵢAᵢ, clip(rᵢ, 1-ε, 1+ε)Aᵢ)] - β·KL(π_θ||π_ref)
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
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

        # Step 4: Compute log probabilities
        outputs = self.model(
            input_ids=generated_ids,
            attention_mask=generated_mask,
        )
        logits = outputs.logits
        logprobs = F.log_softmax(logits, dim=-1)

        # Get log probs of generated tokens
        shift_logprobs = logprobs[:, :-1, :].contiguous()
        shift_labels = generated_ids[:, 1:].contiguous()

        token_logprobs = torch.gather(
            shift_logprobs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        shift_mask = generated_mask[:, 1:].contiguous()
        sequence_logprobs = (token_logprobs * shift_mask).sum(dim=1)

        # Step 5: Compute reference log probs
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
            ref_sequence_logprobs = sequence_logprobs.detach()

        # Step 6: Compute probability ratios
        ratio = torch.exp(sequence_logprobs - ref_sequence_logprobs)

        # Step 7: PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_range,
            1.0 + self.config.clip_range
        ) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # Step 8: KL penalty
        if self.config.kl_coef > 0 and self.reference_model is not None:
            kl_div = sequence_logprobs - ref_sequence_logprobs
            kl_penalty = kl_div.mean()
            total_loss = policy_loss + self.config.kl_coef * kl_penalty
        else:
            kl_penalty = torch.tensor(0.0)
            total_loss = policy_loss

        # Metrics
        metrics = {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_penalty': kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else 0.0,
            'reward_mean': rewards.mean().item(),
            'reward_max': rewards.max().item(),
            'advantage_mean': advantages.mean().item(),
        }

        return total_loss, metrics

    def train(self, train_data: List[Dict]):
        """
        Train the model with GRPO

        Args:
            train_data: List of examples with 'prompt', 'gold_sql', 'database_path'
        """
        print("\n" + "="*60)
        print("🚀 Starting GRPO Training")
        print("="*60)
        print(f"Model: {self.model.config.name_or_path if hasattr(self.model.config, 'name_or_path') else 'Unknown'}")
        print(f"Training examples: {len(train_data)}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Candidates per prompt: {self.config.num_samples_per_prompt}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("="*60)

        self.model.train()

        for epoch in range(self.config.epochs):
            print(f"\n📍 Epoch {epoch + 1}/{self.config.epochs}")
            print("-" * 60)

            epoch_losses = []
            epoch_rewards = []

            progress_bar = tqdm(train_data, desc=f"Training Epoch {epoch+1}")

            for step, example in enumerate(progress_bar):
                # Prepare batch
                prompt = example['prompt']
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=2048
                )

                batch = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask'],
                    'gold_sql': [example['gold_sql']],
                    'database_path': [example.get('database_path', 'test.db')]
                }

                # Compute loss
                loss, metrics = self.compute_loss(batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # Track metrics
                epoch_losses.append(metrics['loss'])
                epoch_rewards.append(metrics['reward_mean'])

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'reward': f"{metrics['reward_mean']:.3f}"
                })

                # Log periodically
                if (step + 1) % self.config.logging_steps == 0:
                    print(f"\n  Step {step+1}/{len(train_data)} | "
                          f"Loss: {metrics['loss']:.4f} | "
                          f"Reward: {metrics['reward_mean']:.3f} | "
                          f"Advantage: {metrics['advantage_mean']:.3f}")

            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)

            print("\n" + "="*60)
            print(f"📊 Epoch {epoch+1} Summary:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Avg Reward: {avg_reward:.3f} ({avg_reward*100:.1f}% accuracy)")
            print("="*60)

            self.training_stats['losses'].append(avg_loss)
            self.training_stats['rewards'].append(avg_reward)

        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)

        # Save model
        self.save_model()

    def save_model(self):
        """Save the trained model"""
        import os
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)

        print(f"\n💾 Model saved to: {self.config.output_dir}")
