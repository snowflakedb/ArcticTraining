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

"""
Attention Isolation Tests for Sample Packing

These tests verify that when multiple samples are packed into a single sequence,
Flash Attention 2 properly isolates them so that:
1. Tokens in Sample B cannot attend to tokens in Sample A
2. Position IDs are correctly reset at sample boundaries
3. Gradients don't leak across sample boundaries

WHY THIS MATTERS:
-----------------
If attention isolation is broken, the model experiences "cross-contamination" where:
- Sample B's hidden states are influenced by Sample A's context
- This degrades model quality, especially for instruction following
- The model may learn spurious correlations between unrelated samples

HOW TO INTERPRET RESULTS:
-------------------------
- test_gradient_leak_isolation: PASS means no gradient flows from Sample B's loss
  to Sample A's tokens. FAIL with non-zero gradients indicates broken isolation.
  
- test_position_ids_reset_correctly: PASS means position IDs reset to 0 at each
  sample boundary. FAIL means samples are treated as one continuous sequence.

- test_needle_in_pack: PASS means the model cannot "see" Sample A's secret from
  Sample B's perspective. FAIL indicates attention leakage.

RUNNING THE TESTS:
------------------
# CPU tests (fast, basic verification)
pytest tests/data/test_attention_isolation.py -v -m "not gpu"

# GPU tests (required for Flash Attention verification)  
pytest tests/data/test_attention_isolation.py -v -m gpu

# All tests with verbose output
pytest tests/data/test_attention_isolation.py -v -s
"""

import pytest
import torch

from arctic_training.data.sft_factory import pack_sft_batch


class TestPositionIdsReset:
    """Tests that verify position IDs are correctly reset at sample boundaries."""

    def test_position_ids_reset_correctly(self):
        """
        Verify that pack_sft_batch correctly resets position IDs at sample boundaries.
        
        WHAT THIS TESTS:
        Position IDs should reset to 0 at the start of each packed sample.
        For example, if we pack [Sample A: 5 tokens] + [Sample B: 3 tokens],
        position_ids should be [0,1,2,3,4, 0,1,2] NOT [0,1,2,3,4,5,6,7].
        
        WHY IT MATTERS:
        Rotary Position Embeddings (RoPE) use position IDs to encode relative
        positions. If Sample B starts at position 5 instead of 0, its positional
        encoding will be wrong, degrading model quality.
        
        PASS: Position IDs reset to 0 at each sample boundary
        FAIL: Position IDs continue incrementing across samples
        """
        # Create mock batch with 3 samples of different lengths
        sample_1 = {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
        sample_2 = {"input_ids": [10, 20, 30], "labels": [-100, 20, 30], "attention_mask": [1, 1, 1]}
        sample_3 = {"input_ids": [100, 200, 300, 400], "labels": [-100, -100, 300, 400], "attention_mask": [1, 1, 1, 1]}

        batch = {
            "input_ids": [sample_1["input_ids"], sample_2["input_ids"], sample_3["input_ids"]],
            "labels": [sample_1["labels"], sample_2["labels"], sample_3["labels"]],
            "attention_mask": [sample_1["attention_mask"], sample_2["attention_mask"], sample_3["attention_mask"]],
        }

        # Pack with max_length large enough to fit all samples
        packed = pack_sft_batch(
            batch,
            max_length=100,
            always_max_length=False,
            drop_last=False,
            fuse_positions_prob=0.0,  # Don't fuse positions for this test
            seed=42,
        )

        # Should produce 1 packed sample containing all 3 original samples
        assert len(packed["position_ids"]) == 1, "Expected 1 packed sample"

        position_ids = packed["position_ids"][0]
        packed_sample_seqlens = packed["packed_sample_seqlens"][0]

        # Verify packed_sample_seqlens matches original lengths
        assert packed_sample_seqlens == [5, 3, 4], f"Expected [5, 3, 4], got {packed_sample_seqlens}"

        # Verify position IDs reset at each boundary
        expected_position_ids = [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 2, 3]
        assert position_ids == expected_position_ids, (
            f"Position IDs should reset at sample boundaries.\n"
            f"Expected: {expected_position_ids}\n"
            f"Got:      {position_ids}"
        )

    def test_packed_sample_seqlens_matches_boundaries(self):
        """
        Verify that packed_sample_seqlens correctly tracks sample boundaries.
        
        WHAT THIS TESTS:
        packed_sample_seqlens should contain the length of each original sample
        in the order they were packed. This is used by the loss computation to
        know where each sample starts and ends.
        
        PASS: packed_sample_seqlens matches actual sample lengths
        FAIL: packed_sample_seqlens doesn't match, causing incorrect loss computation
        """
        sample_1 = {"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1, 1]}
        sample_2 = {"input_ids": [3, 4, 5, 6], "labels": [3, 4, 5, 6], "attention_mask": [1, 1, 1, 1]}

        batch = {
            "input_ids": [sample_1["input_ids"], sample_2["input_ids"]],
            "labels": [sample_1["labels"], sample_2["labels"]],
            "attention_mask": [sample_1["attention_mask"], sample_2["attention_mask"]],
        }

        packed = pack_sft_batch(
            batch,
            max_length=100,
            always_max_length=False,
            drop_last=False,
            fuse_positions_prob=0.0,
            seed=42,
        )

        packed_sample_seqlens = packed["packed_sample_seqlens"][0]

        # Verify lengths match
        assert packed_sample_seqlens == [2, 4], f"Expected [2, 4], got {packed_sample_seqlens}"

        # Verify total length matches
        total_len = sum(packed_sample_seqlens)
        actual_len = len(packed["input_ids"][0])
        assert total_len == actual_len, f"Total seqlens ({total_len}) != actual length ({actual_len})"


class TestGradientIsolation:
    """
    Tests that verify gradients don't leak across sample boundaries.
    
    These tests require a model to run. They verify that when computing loss
    on Sample B, the gradients for Sample A's input embeddings are exactly zero.
    """

    @pytest.fixture
    def small_model_and_tokenizer(self, model_name):
        """Load a small model for gradient testing."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model - for CPU tests, we'll use standard attention
        # For GPU tests with flash_attention_2, use the gpu-marked test
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        model.eval()

        return model, tokenizer

    def test_gradient_leak_isolation_cpu(self, small_model_and_tokenizer):
        """
        Test that gradients don't leak from Sample B's loss to Sample A's embeddings.
        
        WHAT THIS TESTS:
        When we pack [Sample A | Sample B] and compute loss ONLY on Sample B,
        the gradients should not flow back to Sample A's input embeddings.
        If they do, it means Sample B's hidden states depend on Sample A.
        
        HOW IT WORKS:
        1. Create a packed sequence with 2 samples
        2. Set labels so only Sample B contributes to loss
        3. Compute loss and backward pass
        4. Check gradients for Sample A's embedding positions
        
        PASS: Gradients for Sample A positions are exactly 0
        FAIL: Non-zero gradients indicate attention leakage
        
        NOTE: This test uses standard attention on CPU. For Flash Attention 2
        verification, use the GPU-marked test.
        """
        model, tokenizer = small_model_and_tokenizer

        # Sample lengths
        sample_a_len = 10
        sample_b_len = 15

        # Create random token IDs (avoiding special tokens)
        vocab_size = model.config.vocab_size
        sample_a_ids = torch.randint(100, min(1000, vocab_size), (sample_a_len,))
        sample_b_ids = torch.randint(100, min(1000, vocab_size), (sample_b_len,))

        # Pack them together
        packed_input_ids = torch.cat([sample_a_ids, sample_b_ids]).unsqueeze(0)

        # Position IDs with reset at boundary (proper packing)
        position_ids = torch.tensor(
            [list(range(sample_a_len)) + list(range(sample_b_len))]
        )

        # Labels: mask Sample A (-100), train on Sample B
        labels = torch.tensor(
            [[-100] * sample_a_len + sample_b_ids.tolist()]
        )

        # Get input embeddings and enable gradient tracking
        inputs_embeds = model.get_input_embeddings()(packed_input_ids)
        inputs_embeds = inputs_embeds.detach().clone().requires_grad_(True)

        # Forward pass with embeddings
        outputs = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            labels=labels,
        )

        # Backward pass
        outputs.loss.backward()

        # Check gradients for Sample A's positions
        sample_a_grads = inputs_embeds.grad[0, :sample_a_len, :]

        # Gradients should be zero if isolation is working
        max_grad = sample_a_grads.abs().max().item()

        assert max_grad < 1e-6, (
            f"ATTENTION LEAKAGE DETECTED!\n"
            f"Gradients for Sample A should be 0, but max gradient = {max_grad:.6e}\n"
            f"This means Sample B's loss depends on Sample A's embeddings.\n"
            f"Possible causes:\n"
            f"  1. Position IDs are not being used for attention masking\n"
            f"  2. The model doesn't support document-level masking\n"
            f"  3. Flash Attention is not configured for sample isolation"
        )

    @pytest.mark.gpu
    def test_gradient_leak_isolation_flash_attention(self, model_name):
        """
        Test gradient isolation with Flash Attention 2 on GPU.
        
        WHAT THIS TESTS:
        Same as the CPU test, but specifically verifies Flash Attention 2
        behavior with position ID resets.
        
        REQUIRES: GPU with Flash Attention 2 support
        
        PASS: Gradients for Sample A positions are exactly 0
        FAIL: Flash Attention is not properly isolating samples
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Try to load with flash_attention_2
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="cuda",
            )
        except Exception as e:
            pytest.skip(f"Flash Attention 2 not available: {e}")

        model.eval()

        # Sample lengths
        sample_a_len = 10
        sample_b_len = 15

        # Create random token IDs
        vocab_size = model.config.vocab_size
        sample_a_ids = torch.randint(100, min(1000, vocab_size), (sample_a_len,), device="cuda")
        sample_b_ids = torch.randint(100, min(1000, vocab_size), (sample_b_len,), device="cuda")

        # Pack them together
        packed_input_ids = torch.cat([sample_a_ids, sample_b_ids]).unsqueeze(0)

        # Position IDs with reset at boundary
        position_ids = torch.tensor(
            [list(range(sample_a_len)) + list(range(sample_b_len))],
            device="cuda"
        )

        # Labels: mask Sample A, train on Sample B
        labels = torch.tensor(
            [[-100] * sample_a_len + sample_b_ids.tolist()],
            device="cuda"
        )

        # Get embeddings with gradients
        with torch.cuda.amp.autocast(dtype=torch.float16):
            inputs_embeds = model.get_input_embeddings()(packed_input_ids)
            inputs_embeds = inputs_embeds.detach().clone().requires_grad_(True)

            outputs = model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                labels=labels,
            )

            outputs.loss.backward()

        # Check Sample A gradients
        sample_a_grads = inputs_embeds.grad[0, :sample_a_len, :]
        max_grad = sample_a_grads.abs().max().item()

        assert max_grad < 1e-4, (  # Slightly higher tolerance for fp16
            f"FLASH ATTENTION LEAKAGE DETECTED!\n"
            f"Max gradient for Sample A = {max_grad:.6e}\n"
            f"Flash Attention 2 is not properly isolating packed samples.\n"
            f"The model may not support position-ID based document masking."
        )


class TestNeedleInPack:
    """
    Functional tests that verify attention isolation by checking if the model
    can "see" information from Sample A when processing Sample B.
    """

    @pytest.fixture
    def model_and_tokenizer_for_generation(self, model_name):
        """Load model for generation tests."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        model.eval()

        return model, tokenizer

    def test_information_isolation_between_samples(self, model_and_tokenizer_for_generation):
        """
        Test that Sample B cannot access information from Sample A.
        
        WHAT THIS TESTS:
        If we pack a sample containing secret information with a sample asking
        for that information, the model should NOT be able to retrieve it.
        
        NOTE: This test uses a tiny random model, so we can't test actual
        comprehension. Instead, we verify that the hidden states for Sample B
        are identical whether or not Sample A is prepended.
        
        PASS: Hidden states for Sample B are identical with/without Sample A
        FAIL: Hidden states differ, indicating Sample B "sees" Sample A
        """
        model, tokenizer = model_and_tokenizer_for_generation

        # Create two different Sample A options
        sample_a_1 = tokenizer.encode("The secret code is XYZ123.", add_special_tokens=False)
        sample_a_2 = tokenizer.encode("The weather is sunny today.", add_special_tokens=False)
        sample_b = tokenizer.encode("What is the code?", add_special_tokens=False)

        # Pad Sample A options to same length for fair comparison
        max_a_len = max(len(sample_a_1), len(sample_a_2))
        pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id
        sample_a_1 = sample_a_1 + [pad_token] * (max_a_len - len(sample_a_1))
        sample_a_2 = sample_a_2 + [pad_token] * (max_a_len - len(sample_a_2))

        sample_a_len = max_a_len
        sample_b_len = len(sample_b)

        # Create packed sequences
        packed_1 = torch.tensor([sample_a_1 + sample_b])
        packed_2 = torch.tensor([sample_a_2 + sample_b])

        # Position IDs with reset
        position_ids = torch.tensor(
            [list(range(sample_a_len)) + list(range(sample_b_len))]
        )

        # Get hidden states for Sample B portion in both cases
        with torch.no_grad():
            outputs_1 = model(
                input_ids=packed_1,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            outputs_2 = model(
                input_ids=packed_2,
                position_ids=position_ids,
                output_hidden_states=True,
            )

        # Extract last hidden state for Sample B tokens
        hidden_1 = outputs_1.hidden_states[-1][0, sample_a_len:, :]
        hidden_2 = outputs_2.hidden_states[-1][0, sample_a_len:, :]

        # If isolation is working, hidden states should be identical
        # (Sample B can't see Sample A, so changing Sample A shouldn't matter)
        diff = (hidden_1 - hidden_2).abs().max().item()

        # Allow small numerical tolerance
        assert diff < 1e-4, (
            f"INFORMATION LEAKAGE DETECTED!\n"
            f"Hidden states for Sample B differ when Sample A changes.\n"
            f"Max difference: {diff:.6e}\n"
            f"This indicates Sample B can 'see' Sample A through attention.\n"
            f"Position-based attention masking may not be working."
        )


class TestPackingEdgeCases:
    """Tests for edge cases in sample packing."""

    def test_single_sample_packing(self):
        """Verify packing works correctly with a single sample."""
        sample = {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1]}
        batch = {
            "input_ids": [sample["input_ids"]],
            "labels": [sample["labels"]],
            "attention_mask": [sample["attention_mask"]],
        }

        packed = pack_sft_batch(
            batch,
            max_length=100,
            always_max_length=False,
            drop_last=False,
            fuse_positions_prob=0.0,
            seed=42,
        )

        assert len(packed["position_ids"]) == 1
        assert packed["position_ids"][0] == [0, 1, 2]
        assert packed["packed_sample_seqlens"][0] == [3]

    def test_max_length_boundary(self):
        """Verify samples are split correctly at max_length boundary."""
        sample_1 = {"input_ids": [1, 2, 3], "labels": [1, 2, 3], "attention_mask": [1, 1, 1]}
        sample_2 = {"input_ids": [4, 5], "labels": [4, 5], "attention_mask": [1, 1]}
        sample_3 = {"input_ids": [6, 7, 8], "labels": [6, 7, 8], "attention_mask": [1, 1, 1]}

        batch = {
            "input_ids": [sample_1["input_ids"], sample_2["input_ids"], sample_3["input_ids"]],
            "labels": [sample_1["labels"], sample_2["labels"], sample_3["labels"]],
            "attention_mask": [sample_1["attention_mask"], sample_2["attention_mask"], sample_3["attention_mask"]],
        }

        # max_length=5 means sample_1 (3) + sample_2 (2) = 5 fits, sample_3 goes to next
        packed = pack_sft_batch(
            batch,
            max_length=5,
            always_max_length=False,
            drop_last=False,
            fuse_positions_prob=0.0,
            seed=42,
        )

        # Should produce 2 packed samples
        assert len(packed["position_ids"]) == 2, f"Expected 2 packed samples, got {len(packed['position_ids'])}"

        # First pack: samples 1 and 2
        assert packed["packed_sample_seqlens"][0] == [3, 2]
        assert packed["position_ids"][0] == [0, 1, 2, 0, 1]

        # Second pack: sample 3
        assert packed["packed_sample_seqlens"][1] == [3]
        assert packed["position_ids"][1] == [0, 1, 2]

    def test_fuse_positions_disabled(self):
        """Verify fuse_positions_prob=0.0 never fuses positions."""
        samples = [
            {"input_ids": [1, 2], "labels": [1, 2], "attention_mask": [1, 1]},
            {"input_ids": [3, 4, 5], "labels": [3, 4, 5], "attention_mask": [1, 1, 1]},
        ]
        batch = {
            "input_ids": [s["input_ids"] for s in samples],
            "labels": [s["labels"] for s in samples],
            "attention_mask": [s["attention_mask"] for s in samples],
        }

        # Run multiple times to ensure randomness doesn't cause fusion
        for seed in range(10):
            packed = pack_sft_batch(
                batch,
                max_length=100,
                always_max_length=False,
                drop_last=False,
                fuse_positions_prob=0.0,
                seed=seed,
            )

            # Position IDs should always reset
            expected = [0, 1, 0, 1, 2]
            assert packed["position_ids"][0] == expected, (
                f"Seed {seed}: Position IDs should reset but got {packed['position_ids'][0]}"
            )

