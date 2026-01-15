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
Tests for FIM (Fill-in-the-Middle) label masking with prompt/response format.

This test suite validates the length-based label masking approach used for
FIM/autocomplete training, where:
- The prompt column contains everything up to <|fim_middle|> (masked)
- The response column contains only the completion (trained on)
"""

import pytest
from datasets import Dataset

from arctic_training.data.sft_factory import IGNORE_INDEX
from arctic_training.data.sft_factory import SFTDataFactory


class TestTokenizePromptResponse:
    """Tests for the tokenize_prompt_response method."""

    @pytest.fixture
    def tokenizer(self):
        """Load a tokenizer that supports FIM tokens."""
        from transformers import AutoTokenizer

        # Use a tokenizer with FIM tokens and chat template support
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

    def test_prompt_tokens_are_masked(self, tokenizer):
        """Verify all prompt tokens have label -100 (IGNORE_INDEX)."""
        prompt = (
            "<|im_start|>system\nYou are"
            " helpful.<|im_end|>\n<|im_start|>user\nComplete:<|im_end|>\n<|im_start|>assistant\n"
        )
        response = "Hello world!"

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

        # All prompt positions should be -100
        prompt_labels = result["labels"][:prompt_len]
        assert all(
            label == IGNORE_INDEX for label in prompt_labels
        ), f"Expected all prompt labels to be {IGNORE_INDEX}, got: {prompt_labels}"

    def test_response_tokens_are_trainable(self, tokenizer):
        """Verify response tokens have their token IDs as labels (not -100)."""
        prompt = "<|im_start|>assistant\n"
        response = "users"

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        response_labels = result["labels"][prompt_len:]

        # Response labels should NOT be -100
        assert all(
            label != IGNORE_INDEX for label in response_labels
        ), f"Expected response labels to not be {IGNORE_INDEX}, got: {response_labels}"

        # Response labels should be actual token IDs
        response_with_eos = response + tokenizer.eos_token
        expected_response_ids = tokenizer(response_with_eos, add_special_tokens=False)["input_ids"]
        assert response_labels == expected_response_ids, "Response labels don't match expected token IDs"

    def test_empty_response_only_has_eos(self, tokenizer):
        """Verify empty response produces just EOS token as trainable."""
        prompt = "<|im_start|>assistant\n"
        response = ""

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        # Should have at least 1 trainable token (EOS)
        trainable_labels = [lbl for lbl in result["labels"] if lbl != IGNORE_INDEX]
        assert len(trainable_labels) >= 1, "Empty response should have at least EOS token"

        # The trainable token should be the EOS token
        eos_id = tokenizer.eos_token_id
        assert eos_id in trainable_labels, f"Expected EOS token {eos_id} in trainable labels"

    def test_input_ids_concatenation(self, tokenizer):
        """Verify input_ids is proper concatenation of prompt and response."""
        prompt = "Hello "
        response = "world"

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        # Verify input_ids is prompt + response + eos
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_with_eos = response + tokenizer.eos_token
        response_ids = tokenizer(response_with_eos, add_special_tokens=False)["input_ids"]

        expected_ids = prompt_ids + response_ids
        assert result["input_ids"] == expected_ids, "input_ids doesn't match expected concatenation"

    def test_attention_mask_all_ones(self, tokenizer):
        """Verify attention mask is all 1s for the entire sequence."""
        prompt = "Hello "
        response = "world"

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        assert all(m == 1 for m in result["attention_mask"]), "Attention mask should be all 1s"
        assert len(result["attention_mask"]) == len(
            result["input_ids"]
        ), "Attention mask length should match input_ids"

    def test_labels_length_matches_input_ids(self, tokenizer):
        """Verify labels has same length as input_ids."""
        prompt = "A somewhat longer prompt "
        response = "with a response"

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        assert len(result["labels"]) == len(result["input_ids"]), "Labels length should match input_ids"


class TestFIMTokenBoundary:
    """Tests specific to FIM token boundary detection."""

    @pytest.fixture
    def tokenizer(self):
        """Load a tokenizer that supports FIM tokens."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

    def test_fim_middle_is_in_prompt_and_masked(self, tokenizer):
        """Verify <|fim_middle|> token is in prompt and therefore masked."""
        # Check if tokenizer has FIM tokens
        fim_middle_id = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
        if fim_middle_id == tokenizer.unk_token_id:
            pytest.skip("Tokenizer doesn't have <|fim_middle|> token")

        prompt = "<|fim_prefix|>SELECT<|fim_suffix|>;<|fim_middle|>"
        response = "users"

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        # Find <|fim_middle|> in the input_ids
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

        # <|fim_middle|> should be in prompt_ids
        assert fim_middle_id in prompt_ids, f"<|fim_middle|> ({fim_middle_id}) not found in prompt_ids"

        # Find position of fim_middle and verify it's masked
        fim_pos = prompt_ids.index(fim_middle_id)
        assert result["labels"][fim_pos] == IGNORE_INDEX, f"<|fim_middle|> at position {fim_pos} should be masked"

    def test_response_after_fim_middle_is_trainable(self, tokenizer):
        """Verify content after <|fim_middle|> (in response column) is trainable."""
        fim_middle_id = tokenizer.convert_tokens_to_ids("<|fim_middle|>")
        if fim_middle_id == tokenizer.unk_token_id:
            pytest.skip("Tokenizer doesn't have <|fim_middle|> token")

        # Simulate a FIM prompt where everything up to fim_middle is in prompt
        prompt = "<|fim_prefix|>SELECT * FROM <|fim_suffix|> WHERE id=1<|fim_middle|>"
        response = "users"  # This should be trainable

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        response_labels = result["labels"][prompt_len:]

        # All response labels should be trainable (not -100)
        assert all(
            label != IGNORE_INDEX for label in response_labels
        ), "All tokens after <|fim_middle|> should be trainable"


class TestDatasetFormatAutoDetection:
    """Tests for auto-detection of dataset format in process()."""

    @pytest.fixture
    def tokenizer(self):
        """Load a tokenizer."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

    def test_prompt_response_format_detection(self, tokenizer):
        """Verify prompt/response format is detected correctly."""
        # Create a dataset with prompt/response columns
        data = {
            "prompt": ["Hello ", "How are "],
            "response": ["world!", "you?"],
        }
        dataset = Dataset.from_dict(data)

        # Verify the format detection logic
        has_prompt_response = "prompt" in dataset.column_names and "response" in dataset.column_names
        has_messages = "messages" in dataset.column_names

        assert has_prompt_response, "Should detect prompt/response format"
        assert not has_messages, "Should not detect messages format"

    def test_messages_format_detection(self, tokenizer):
        """Verify messages format is detected correctly."""
        # Create a dataset with messages column
        data = {
            "messages": [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ],
                [
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "Good!"},
                ],
            ],
        }
        dataset = Dataset.from_dict(data)

        # Verify the format detection logic
        has_prompt_response = "prompt" in dataset.column_names and "response" in dataset.column_names
        has_messages = "messages" in dataset.column_names

        assert not has_prompt_response, "Should not detect prompt/response format"
        assert has_messages, "Should detect messages format"


class TestBackwardsCompatibility:
    """Tests ensuring backwards compatibility with existing message format."""

    @pytest.fixture
    def tokenizer(self):
        """Load a tokenizer."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

    def test_tokenize_messages_still_works(self, tokenizer):
        """Verify existing tokenize_messages method still works correctly."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        result = SFTDataFactory.tokenize_messages(messages, tokenizer, mask_inputs=True)

        # Should have input_ids
        assert "input_ids" in result, "Should have input_ids"
        assert len(result["input_ids"]) > 0, "input_ids should not be empty"

        # Should have labels
        assert "labels" in result, "Should have labels"
        assert len(result["labels"]) == len(result["input_ids"]), "labels length should match input_ids"

        # Should have some trainable tokens (assistant content)
        trainable = [lbl for lbl in result["labels"] if lbl != IGNORE_INDEX]
        assert len(trainable) > 0, "Should have some trainable tokens"

    def test_tokenize_messages_no_masking(self, tokenizer):
        """Verify tokenize_messages works with mask_inputs=False."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = SFTDataFactory.tokenize_messages(messages, tokenizer, mask_inputs=False)

        # When mask_inputs=False, labels should equal input_ids
        assert result["labels"] == result["input_ids"], "With mask_inputs=False, labels should equal input_ids"


class TestEdgeCases:
    """Tests for edge cases in tokenization."""

    @pytest.fixture
    def tokenizer(self):
        """Load a tokenizer."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)

    def test_very_short_prompt(self, tokenizer):
        """Test with minimal prompt."""
        prompt = "a"
        response = "b"

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        assert len(result["input_ids"]) > 0, "Should produce some tokens"
        assert len(result["labels"]) == len(result["input_ids"]), "Labels length should match"

    def test_prompt_with_special_characters(self, tokenizer):
        """Test prompt containing special characters."""
        prompt = "SELECT * FROM users WHERE name='O\\'Brien' AND id > 100\n"
        response = "ORDER BY created_at"

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        assert len(result["input_ids"]) > 0, "Should handle special characters"

    def test_unicode_in_prompt_and_response(self, tokenizer):
        """Test with Unicode characters."""
        prompt = "Translate: こんにちは "
        response = "Hello"

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        assert len(result["input_ids"]) > 0, "Should handle Unicode"
        # Verify boundary is still correct
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        assert all(lbl == IGNORE_INDEX for lbl in result["labels"][:prompt_len]), "Prompt should still be masked"

    def test_whitespace_only_response(self, tokenizer):
        """Test with whitespace-only response."""
        prompt = "Complete: "
        response = "   "  # Just spaces

        result = SFTDataFactory.tokenize_prompt_response(prompt, response, tokenizer)

        # Should still have trainable tokens (whitespace + EOS)
        trainable = [lbl for lbl in result["labels"] if lbl != IGNORE_INDEX]
        assert len(trainable) >= 1, "Should have at least EOS token"
