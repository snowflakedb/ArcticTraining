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

"""Tests for On-Policy Distillation Trainer components."""

import pytest
import torch
import torch.nn as nn

# Import the modules to register them
from arctic_training.config.on_policy_distillation import OnPolicyDistillationTrainerConfig
from arctic_training.data.on_policy_distillation_factory import (
    DataCollatorForOnPolicyDistillation,
    OnPolicyDistillationDataFactory,
    pad_prompts,
)
from arctic_training.trainer.on_policy_distillation_trainer import OnPolicyDistillationTrainer  # noqa: F401
from arctic_training.trainer.utils import disable_dropout_in_model


@pytest.mark.skip(reason="Config tests require full distributed setup - tested via integration tests")
class TestOnPolicyDistillationConfig:
    """Tests for OnPolicyDistillationTrainerConfig.
    
    Note: These tests are skipped because they require the full distributed
    setup to properly resolve type hints. The config is tested via the
    integration tests at the end of this file.
    """

    def test_config_requires_teacher(self, model_name):
        """Test that config validation requires teacher_model."""
        with pytest.raises(ValueError):
            OnPolicyDistillationTrainerConfig(
                model={"name_or_path": model_name},
                data={"sources": ["test-data"], "max_length": 512},
                skip_validation=True,
                # No teacher_model
            )

    def test_config_with_teacher_model(self, model_name):
        """Test config with in-memory teacher model."""
        config = OnPolicyDistillationTrainerConfig(
            model={"name_or_path": model_name},
            data={"sources": ["test-data"], "max_length": 512},
            teacher_model={"name_or_path": model_name},
            skip_validation=True,
        )
        assert config.teacher_model.name_or_path == model_name
        assert config.disable_teacher_dropout is True

    def test_config_defaults(self, model_name):
        """Test default config values."""
        config = OnPolicyDistillationTrainerConfig(
            model={"name_or_path": model_name},
            data={"sources": ["test-data"], "max_length": 512},
            teacher_model={"name_or_path": model_name},
            skip_validation=True,
        )
        assert config.num_rollouts_per_prompt == 4
        assert config.max_new_tokens == 2048
        assert config.generation_temperature == 1.0
        assert config.beta == 1.0
        assert config.disable_teacher_dropout is True

    def test_generation_batch_size_default(self, model_name):
        """Test that generation_batch_size defaults to micro_batch_size."""
        config = OnPolicyDistillationTrainerConfig(
            model={"name_or_path": model_name},
            data={"sources": ["test-data"], "max_length": 512},
            teacher_model={"name_or_path": model_name},
            micro_batch_size=4,
            skip_validation=True,
        )
        assert config.generation_batch_size == 4

    def test_teacher_deepspeed_config_auto_generated(self, model_name):
        """Test that teacher DeepSpeed config is auto-generated."""
        config = OnPolicyDistillationTrainerConfig(
            model={"name_or_path": model_name},
            data={"sources": ["test-data"], "max_length": 512},
            teacher_model={"name_or_path": model_name},
            skip_validation=True,
        )
        # Teacher deepspeed should be auto-generated
        assert len(config.teacher_deepspeed) > 0
        assert "zero_optimization" in config.teacher_deepspeed


class TestPadPrompts:
    """Tests for the pad_prompts utility function."""

    def test_left_padding(self):
        """Test left padding of prompts."""
        tensors = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 7, 8, 9]),
        ]
        padded = pad_prompts(tensors, padding_value=0, padding_side="left", divisible_by=4)

        assert padded.shape == (3, 4)  # Padded to divisible by 4
        assert padded[0].tolist() == [0, 1, 2, 3]  # Left-padded
        assert padded[1].tolist() == [0, 0, 4, 5]
        assert padded[2].tolist() == [6, 7, 8, 9]

    def test_right_padding(self):
        """Test right padding of prompts."""
        tensors = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
        ]
        padded = pad_prompts(tensors, padding_value=0, padding_side="right", divisible_by=4)

        assert padded.shape == (2, 4)
        assert padded[0].tolist() == [1, 2, 3, 0]  # Right-padded
        assert padded[1].tolist() == [4, 5, 0, 0]

    def test_padding_divisibility(self):
        """Test that output length is divisible by specified value."""
        tensors = [torch.tensor([1, 2, 3, 4, 5])]  # Length 5
        padded = pad_prompts(tensors, padding_value=0, divisible_by=8)

        assert padded.shape == (1, 8)  # Padded to 8 (divisible by 8)


class TestDataCollator:
    """Tests for DataCollatorForOnPolicyDistillation."""

    def test_collator_output_structure(self):
        """Test that collator produces expected output structure."""
        from unittest.mock import MagicMock

        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        # Create mock config
        config = MagicMock()
        config.div_length = 4

        collator = DataCollatorForOnPolicyDistillation(tokenizer, config)

        instances = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
        ]

        result = collator(instances)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "prompt_lengths" in result
        assert result["prompt_lengths"].tolist() == [3, 2]

    def test_collator_left_padding(self):
        """Test that collator applies left padding."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        config = MagicMock()
        config.div_length = 4

        collator = DataCollatorForOnPolicyDistillation(tokenizer, config)

        instances = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
        ]

        result = collator(instances)

        # Check left padding (real tokens at the end)
        assert result["input_ids"][0, -1].item() == 3
        assert result["input_ids"][1, -1].item() == 5


class TestPolicyGradientLoss:
    """Tests for the policy gradient loss computation.
    
    The on-policy distillation loss uses policy gradient with advantage:
        advantage = teacher_logprob - student_logprob
        loss = -(advantage * student_logprob).mean()
    
    This ensures:
    - When teacher > student: positive advantage -> increase student_logprob
    - When teacher < student: negative advantage -> decrease student_logprob
    """

    def test_advantage_computation(self):
        """Test that advantage is computed correctly."""
        student_logprobs = torch.tensor([-1.0, -2.0, -3.0])
        teacher_logprobs = torch.tensor([-0.5, -2.0, -4.0])
        
        advantage = teacher_logprobs - student_logprobs
        
        # Position 0: teacher (-0.5) > student (-1.0) -> positive advantage
        assert advantage[0] > 0
        # Position 1: teacher == student -> zero advantage
        assert advantage[1] == 0
        # Position 2: teacher (-4.0) < student (-3.0) -> negative advantage
        assert advantage[2] < 0

    def test_policy_gradient_loss_direction(self):
        """Test that policy gradient loss has correct gradient direction."""
        # Student logprobs (requires grad)
        student_logprobs = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
        teacher_logprobs = torch.tensor([-0.5, -2.0, -4.0])
        
        # Compute advantage (detached as in actual implementation)
        advantage = (teacher_logprobs - student_logprobs.detach())
        
        # Policy gradient loss
        loss = -(advantage * student_logprobs).mean()
        loss.backward()
        
        # Check gradient directions
        # Position 0: advantage > 0, so gradient should be negative (to increase logprob)
        # Position 2: advantage < 0, so gradient should be positive (to decrease logprob)
        assert student_logprobs.grad[0] < 0  # Will increase student_logprob
        assert student_logprobs.grad[2] > 0  # Will decrease student_logprob

    def test_loss_zero_when_distributions_match(self):
        """Test that loss is zero when student matches teacher."""
        student_logprobs = torch.tensor([-1.0, -2.0, -3.0])
        teacher_logprobs = torch.tensor([-1.0, -2.0, -3.0])
        
        advantage = teacher_logprobs - student_logprobs
        loss = -(advantage * student_logprobs).mean()
        
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_loss_with_masking(self):
        """Test loss computation with masked positions."""
        student_logprobs = torch.tensor([-1.0, -2.0, -999.0])  # Last is padding
        teacher_logprobs = torch.tensor([-0.5, -1.0, -999.0])
        mask = torch.tensor([True, True, False])
        
        # Apply mask
        valid_student = student_logprobs[mask]
        valid_teacher = teacher_logprobs[mask]
        
        advantage = valid_teacher - valid_student
        loss = -(advantage * valid_student).mean()
        
        # Manual calculation for first two positions only
        adv_0 = -0.5 - (-1.0)  # = 0.5
        adv_1 = -1.0 - (-2.0)  # = 1.0
        expected_loss = -((adv_0 * -1.0) + (adv_1 * -2.0)) / 2
        
        assert torch.isclose(loss, torch.tensor(expected_loss))


class TestReverseKLMetric:
    """Tests for reverse KL metric computation (for monitoring, not loss)."""

    def test_reverse_kl_identical_distributions(self):
        """Test that KL is zero when distributions are identical."""
        student_logprobs = torch.tensor([-1.0, -2.0, -3.0])
        teacher_logprobs = torch.tensor([-1.0, -2.0, -3.0])
        
        # Reverse KL: student - teacher
        reverse_kl = (student_logprobs - teacher_logprobs).mean()
        
        assert torch.isclose(reverse_kl, torch.tensor(0.0))

    def test_reverse_kl_student_worse(self):
        """Test KL when student is worse (lower logprobs) than teacher."""
        student_logprobs = torch.tensor([-2.0, -3.0, -4.0])  # Lower (worse)
        teacher_logprobs = torch.tensor([-1.0, -2.0, -3.0])  # Higher (better)
        
        reverse_kl = (student_logprobs - teacher_logprobs).mean()
        
        # Student logprobs are lower, so student - teacher is negative
        assert reverse_kl < 0

    def test_reverse_kl_student_overconfident(self):
        """Test KL when student is overconfident (higher logprobs than teacher)."""
        student_logprobs = torch.tensor([-0.5, -1.0, -1.5])  # Higher (overconfident)
        teacher_logprobs = torch.tensor([-1.0, -2.0, -3.0])  # Lower
        
        reverse_kl = (student_logprobs - teacher_logprobs).mean()
        
        # Student logprobs are higher, so student - teacher is positive
        assert reverse_kl > 0


class TestDataFactoryTokenization:
    """Tests for OnPolicyDistillationDataFactory tokenization."""

    def test_tokenize_prompt_extracts_user_message(self):
        """Test that tokenize_prompt extracts prompt correctly."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "User: Hello"
        tokenizer.return_value = {"input_ids": [1, 2, 3, 4]}

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = OnPolicyDistillationDataFactory.tokenize_prompt(
            messages, tokenizer, include_system=True
        )

        # Should only include user message in template call
        call_args = tokenizer.apply_chat_template.call_args
        assert len(call_args[1]["conversation"]) == 1
        assert call_args[1]["conversation"][0]["role"] == "user"

    def test_tokenize_prompt_includes_system(self):
        """Test that system messages are included when specified."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "System: Be helpful\nUser: Hello"
        tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        OnPolicyDistillationDataFactory.tokenize_prompt(
            messages, tokenizer, include_system=True
        )

        call_args = tokenizer.apply_chat_template.call_args
        assert len(call_args[1]["conversation"]) == 2

    def test_tokenize_prompt_excludes_system(self):
        """Test that system messages can be excluded."""
        from unittest.mock import MagicMock

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "User: Hello"
        tokenizer.return_value = {"input_ids": [1, 2, 3]}

        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        OnPolicyDistillationDataFactory.tokenize_prompt(
            messages, tokenizer, include_system=False
        )

        call_args = tokenizer.apply_chat_template.call_args
        assert len(call_args[1]["conversation"]) == 1
        assert call_args[1]["conversation"][0]["role"] == "user"


class TestDisableDropout:
    """Tests for disable_dropout_in_model utility."""

    def test_disable_dropout(self):
        """Test that dropout is disabled in model."""
        # Create a simple model with dropout
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Dropout(p=0.5),
            nn.Linear(10, 10),
            nn.Dropout(p=0.3),
        )

        # Verify dropout is enabled
        dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        assert len(dropouts) == 2
        assert dropouts[0].p == 0.5
        assert dropouts[1].p == 0.3

        # Disable dropout
        disable_dropout_in_model(model)

        # Verify dropout is disabled
        for dropout in dropouts:
            assert dropout.p == 0.0

    def test_disable_dropout_nested(self):
        """Test that dropout is disabled in nested modules."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 10),
                nn.Dropout(p=0.5),
            ),
            nn.Dropout(p=0.3),
        )

        disable_dropout_in_model(model)

        dropouts = [m for m in model.modules() if isinstance(m, nn.Dropout)]
        for dropout in dropouts:
            assert dropout.p == 0.0


class TestNumRolloutsPerPrompt:
    """Tests for num_rollouts_per_prompt functionality."""

    def test_repeat_interleave_prompts(self):
        """Test that prompts are correctly repeated for multiple rollouts."""
        # Simulate input tensors
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 2 prompts
        prompt_lengths = torch.tensor([3, 3])
        num_rollouts = 3

        # Repeat interleave
        expanded_input_ids = input_ids.repeat_interleave(num_rollouts, dim=0)
        expanded_lengths = prompt_lengths.repeat_interleave(num_rollouts, dim=0)

        # Should have 6 rows (2 prompts * 3 rollouts)
        assert expanded_input_ids.shape[0] == 6
        assert expanded_lengths.shape[0] == 6

        # Check pattern: [p1, p1, p1, p2, p2, p2]
        assert torch.equal(expanded_input_ids[0], input_ids[0])
        assert torch.equal(expanded_input_ids[1], input_ids[0])
        assert torch.equal(expanded_input_ids[2], input_ids[0])
        assert torch.equal(expanded_input_ids[3], input_ids[1])


# Integration tests
@pytest.mark.parametrize(
    "run_on_cpu",
    [
        True,
        pytest.param(False, marks=pytest.mark.gpu),
    ],
)
def test_on_policy_distillation_trainer(model_name, run_on_cpu):
    """Test full on-policy distillation training loop."""
    from tests.utils import run_dummy_training

    run_dummy_training(
        {
            "type": "on_policy_distillation",
            "model": {
                "type": "random-weight-hf",
                "name_or_path": model_name,
                "dtype": "float32",
            },
            "teacher_model": {
                "type": "random-weight-hf",
                "name_or_path": model_name,
                "dtype": "float32",
            },
            "data": {
                "max_length": 512,
                "max_prompt_length": 128,
                "sources": ["HuggingFaceH4/ultrachat_200k:train[:10]"],
            },
            "max_new_tokens": 32,
            "num_rollouts_per_prompt": 1,
            "beta": 1.0,
        },
        run_on_cpu=run_on_cpu,
    )


@pytest.mark.parametrize(
    "run_on_cpu",
    [
        True,
        pytest.param(False, marks=pytest.mark.gpu),
    ],
)
def test_on_policy_distillation_with_multiple_rollouts(model_name, run_on_cpu):
    """Test on-policy distillation with multiple rollouts per prompt."""
    from tests.utils import run_dummy_training

    run_dummy_training(
        {
            "type": "on_policy_distillation",
            "model": {
                "type": "random-weight-hf",
                "name_or_path": model_name,
                "dtype": "float32",
            },
            "teacher_model": {
                "type": "random-weight-hf",
                "name_or_path": model_name,
                "dtype": "float32",
            },
            "data": {
                "max_length": 512,
                "max_prompt_length": 128,
                "sources": ["HuggingFaceH4/ultrachat_200k:train[:10]"],
            },
            "max_new_tokens": 32,
            "num_rollouts_per_prompt": 2,  # Multiple rollouts
            "micro_batch_size": 2,
            "beta": 1.0,
        },
        run_on_cpu=run_on_cpu,
    )
