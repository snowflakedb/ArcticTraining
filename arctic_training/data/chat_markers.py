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
Chat template marker detection and token-based label masking.

This module provides utilities for detecting chat template markers (e.g., user/assistant
turn boundaries) and creating labels for SFT training using token-based matching rather
than character position matching. This approach:

1. Correctly includes end-of-turn tokens in the training signal
2. Is more robust to tokenizer variations
3. Works with any chat template format

References:
- Unsloth's train_on_responses_only: https://github.com/unslothai/unsloth-zoo
"""

import re
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

from transformers import PreTrainedTokenizerBase

from arctic_training.logging import logger

IGNORE_INDEX = -100


@dataclass
class ChatMarkerConfig:
    """Markers for identifying role boundaries in chat templates."""

    user_start: str  # Marks start of user turn
    assistant_start: str  # Marks start of assistant turn
    turn_end: str  # Marks end of any turn
    system_start: Optional[str] = None  # Optional system marker


# Pre-defined configurations for popular model families
KNOWN_CHAT_MARKERS: Dict[str, ChatMarkerConfig] = {
    # ===== ChatML Format (Qwen, Yi, Arctic, etc.) =====
    "chatml": ChatMarkerConfig(
        user_start="<|im_start|>user\n",
        assistant_start="<|im_start|>assistant\n",
        turn_end="<|im_end|>",
        system_start="<|im_start|>system\n",
    ),
    # ===== Llama 3 / 3.1 / 3.2 =====
    "llama3": ChatMarkerConfig(
        user_start="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant_start="<|start_header_id|>assistant<|end_header_id|>\n\n",
        turn_end="<|eot_id|>",
        system_start="<|start_header_id|>system<|end_header_id|>\n\n",
    ),
    # ===== Llama 2 / Mistral (Instruct format) =====
    "llama2": ChatMarkerConfig(
        user_start="[INST] ",
        assistant_start=" [/INST] ",
        turn_end="</s>",
        system_start="<<SYS>>\n",
    ),
    # ===== Mistral v0.3+ / Mixtral =====
    "mistral_v3": ChatMarkerConfig(
        user_start="[INST]",
        assistant_start="[/INST]",
        turn_end="</s>",
    ),
    # ===== Phi-3 / Phi-4 =====
    "phi3": ChatMarkerConfig(
        user_start="<|user|>\n",
        assistant_start="<|assistant|>\n",
        turn_end="<|end|>",
        system_start="<|system|>\n",
    ),
    # ===== Gemma / Gemma 2 =====
    "gemma": ChatMarkerConfig(
        user_start="<start_of_turn>user\n",
        assistant_start="<start_of_turn>model\n",
        turn_end="<end_of_turn>",
    ),
    # ===== DeepSeek =====
    "deepseek": ChatMarkerConfig(
        user_start="User: ",
        assistant_start="Assistant: ",
        turn_end="<｜end▁of▁sentence｜>",
    ),
    # ===== DeepSeek V2/V3 (ChatML-like) =====
    "deepseek_v2": ChatMarkerConfig(
        user_start="<|User|>",
        assistant_start="<|Assistant|>",
        turn_end="<|end_of_sentence|>",
    ),
    # ===== Vicuna / Alpaca =====
    "vicuna": ChatMarkerConfig(
        user_start="USER: ",
        assistant_start="ASSISTANT: ",
        turn_end="</s>",
    ),
    # ===== Zephyr (based on Mistral) =====
    "zephyr": ChatMarkerConfig(
        user_start="<|user|>\n",
        assistant_start="<|assistant|>\n",
        turn_end="</s>",
        system_start="<|system|>\n",
    ),
    # ===== Command-R (Cohere) =====
    "command_r": ChatMarkerConfig(
        user_start="<|START_OF_TURN_TOKEN|><|USER_TOKEN|>",
        assistant_start="<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
        turn_end="<|END_OF_TURN_TOKEN|>",
        system_start="<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>",
    ),
}


def detect_model_family(tokenizer: PreTrainedTokenizerBase) -> Optional[str]:
    """
    Detect which model family the tokenizer belongs to.
    Returns the key for KNOWN_CHAT_MARKERS or None if unknown.
    """
    # Check by model name patterns
    model_name = getattr(tokenizer, "name_or_path", "").lower()

    patterns = [
        (r"qwen|arctic", "chatml"),
        (r"llama-3|llama3|llama-4|llama4", "llama3"),
        (r"llama-2|llama2", "llama2"),
        (r"mistral.*v0\.[3-9]|mixtral", "mistral_v3"),
        (r"phi-[34]|phi[34]", "phi3"),
        (r"gemma", "gemma"),
        (r"deepseek.*v[23]|deepseek-v[23]", "deepseek_v2"),
        (r"deepseek", "deepseek"),
        (r"vicuna|alpaca", "vicuna"),
        (r"zephyr", "zephyr"),
        (r"command-r|c4ai", "command_r"),
    ]

    for pattern, family in patterns:
        if re.search(pattern, model_name):
            logger.info(f"Detected model family '{family}' from model name: {model_name}")
            return family

    # Check by special tokens
    special_tokens = set(getattr(tokenizer, "additional_special_tokens", []))

    if "<|im_start|>" in special_tokens:
        logger.info("Detected model family 'chatml' from special tokens")
        return "chatml"
    if "<|start_header_id|>" in special_tokens:
        logger.info("Detected model family 'llama3' from special tokens")
        return "llama3"
    if "<start_of_turn>" in special_tokens:
        logger.info("Detected model family 'gemma' from special tokens")
        return "gemma"
    if "<|user|>" in special_tokens and "<|assistant|>" in special_tokens:
        logger.info("Detected model family 'phi3' from special tokens")
        return "phi3"

    return None  # Unknown - will use heuristic


def detect_markers_heuristic(tokenizer: PreTrainedTokenizerBase) -> ChatMarkerConfig:
    """
    Auto-detect chat markers by applying the template to minimal examples and diffing.
    This is the fallback when the model family is unknown.
    """
    # Minimal conversations with unique single-char content
    user_only = [{"role": "user", "content": "X"}]
    user_asst = [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}]
    multi_turn = [
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": "Y"},
        {"role": "user", "content": "Z"},
    ]

    try:
        t1 = tokenizer.apply_chat_template(user_only, tokenize=False, add_generation_prompt=False)
        t2 = tokenizer.apply_chat_template(user_asst, tokenize=False, add_generation_prompt=False)
        t3 = tokenizer.apply_chat_template(multi_turn, tokenize=False, add_generation_prompt=False)
    except Exception as e:
        logger.warning(f"Failed to apply chat template for heuristic detection: {e}")
        # Return a safe default that won't match anything - will fall back to char-based
        return ChatMarkerConfig(
            user_start="<|UNKNOWN_USER|>",
            assistant_start="<|UNKNOWN_ASSISTANT|>",
            turn_end="<|UNKNOWN_END|>",
        )

    # Extract markers by diffing
    # User turn = t1 (contains user start + "X" + turn end + maybe newline)
    # Find where X appears
    x_pos = t1.find("X")
    user_start = t1[:x_pos] if x_pos > 0 else ""

    # Assistant turn = difference between t2 and t1
    # t2 should start with t1's content (possibly without trailing generation prompt)
    if t2.startswith(t1):
        asst_turn = t2[len(t1) :]
    else:
        # Fallback: find Y in t2 and extract the marker before it
        # This avoids including the previous turn's end marker
        y_pos_t2 = t2.find("Y")
        x_pos_t2 = t2.find("X")
        if y_pos_t2 > x_pos_t2 >= 0:
            # Extract from after X's position to Y, then find where assistant marker starts
            # Look for common assistant markers in the segment between X and Y
            segment = t2[x_pos_t2 + 1 : y_pos_t2]
            # Find where a new role marker likely starts (after any turn end token)
            for end_marker in ["<|im_end|>", "<|eot_id|>", "</s>", "<|end|>", "<end_of_turn>"]:
                if end_marker in segment:
                    # Assistant start is everything after the end marker (plus any newline)
                    marker_end = segment.find(end_marker) + len(end_marker)
                    asst_turn = segment[marker_end:].lstrip("\n") + "Y"
                    break
            else:
                # No known end marker found, use original fallback
                asst_turn = segment + "Y"
        else:
            asst_turn = t2[t2.find("X") + 1 :]

    # Find Y in assistant turn
    y_pos_in_asst = asst_turn.find("Y")
    assistant_start = asst_turn[:y_pos_in_asst] if y_pos_in_asst > 0 else ""

    # Turn end = what comes after Y and before the next user turn in t3
    # Find where Y ends and Z's user turn starts
    y_pos_t3 = t3.find("Y")
    z_pos_t3 = t3.find("Z")
    if y_pos_t3 >= 0 and z_pos_t3 > y_pos_t3:
        between = t3[y_pos_t3 + 1 : z_pos_t3]
        # Turn end is the part before the next user_start
        if user_start and user_start in between:
            turn_end = between[: between.find(user_start)]
        else:
            # Just take up to some reasonable delimiter
            turn_end = between.split("\n")[0] if "\n" in between else between
    else:
        # Fallback: look for common end tokens
        turn_end = ""
        for common_end in ["<|im_end|>", "<|eot_id|>", "</s>", "<|end|>", "<end_of_turn>"]:
            if common_end in t2:
                turn_end = common_end
                break

    # Clean up extracted markers
    user_start = user_start.strip() if user_start else "<|user|>"
    assistant_start = assistant_start.strip() if assistant_start else "<|assistant|>"
    turn_end = turn_end.strip() if turn_end else ""

    if not turn_end:
        logger.warning(
            "Could not detect turn_end marker from chat template. "
            "Assistant response boundaries may not be correctly identified. "
            "Consider explicitly specifying chat_template_family in your config."
        )

    logger.info(
        "Heuristically detected chat markers:\n"
        f"  user_start: {repr(user_start)}\n"
        f"  assistant_start: {repr(assistant_start)}\n"
        f"  turn_end: {repr(turn_end)}"
    )

    return ChatMarkerConfig(
        user_start=user_start,
        assistant_start=assistant_start,
        turn_end=turn_end,
    )


def get_chat_markers(
    tokenizer: PreTrainedTokenizerBase,
    chat_template_family: Optional[str] = None,
) -> ChatMarkerConfig:
    """
    Get chat markers for the given tokenizer.
    Priority: explicit family > detected family > heuristic detection
    """
    # 1. Explicit family specified
    if chat_template_family is not None:
        if chat_template_family in KNOWN_CHAT_MARKERS:
            logger.info(f"Using explicitly specified chat template family: {chat_template_family}")
            return KNOWN_CHAT_MARKERS[chat_template_family]
        else:
            raise ValueError(
                f"Unknown chat_template_family: {chat_template_family}. "
                f"Available options: {list(KNOWN_CHAT_MARKERS.keys())}"
            )

    # 2. Try to detect model family
    family = detect_model_family(tokenizer)
    if family is not None:
        return KNOWN_CHAT_MARKERS[family]

    # 3. Heuristic detection (fallback)
    logger.info("Model family not recognized, using heuristic marker detection")
    return detect_markers_heuristic(tokenizer)


def _find_subsequence(sequence: List[int], subsequence: List[int], start: int = 0) -> int:
    """
    Find the starting index of a subsequence within a sequence.
    Returns -1 if not found or if subsequence is empty.
    """
    if not subsequence:
        return -1  # Empty subsequence is considered "not found"
    subseq_len = len(subsequence)
    for i in range(start, len(sequence) - subseq_len + 1):
        if sequence[i : i + subseq_len] == subsequence:
            return i
    return -1


def _tokenize_marker(marker: str, tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """
    Tokenize a marker string, handling tokenizer variations.
    Similar to Unsloth's _find_common_token_ids but simplified.
    """
    if not marker:
        return []
    return tokenizer(marker, add_special_tokens=False).input_ids


def _tokenize_marker_without_trailing_whitespace(marker: str, tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """
    Tokenize a marker string, stripping trailing whitespace to avoid tokenizer
    context sensitivity issues.

    Some tokenizers (e.g., Qwen) tokenize "\\n\\n" differently than "\\n" + "\\n".
    When assistant content starts with "\\n", the marker "assistant\\n" + "\\n" becomes
    "assistant\\n\\n" which tokenizes differently, causing pattern match failures.

    By stripping trailing whitespace from the marker, we avoid this issue.
    """
    if not marker:
        return []
    # Strip trailing whitespace (newlines, spaces) to avoid context sensitivity
    stripped = marker.rstrip()
    return tokenizer(stripped, add_special_tokens=False).input_ids


def get_token_based_labels(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizerBase,
    markers: ChatMarkerConfig,
) -> List[int]:
    """
    Create labels using token-based pattern matching.

    This approach:
    1. Finds assistant_start marker tokens in input_ids
    2. Marks tokens as trainable from assistant_start until turn_end or next user_start
    3. Includes the turn_end token in the training signal

    Args:
        input_ids: The tokenized input sequence
        tokenizer: The tokenizer used
        markers: Chat template markers

    Returns:
        List of labels where trainable tokens have their token ID and masked tokens have -100
    """
    # Tokenize the markers - strip trailing whitespace to avoid context sensitivity
    # This handles cases where content starts with newlines causing different tokenization
    user_start_ids = _tokenize_marker_without_trailing_whitespace(markers.user_start, tokenizer)
    assistant_start_ids = _tokenize_marker_without_trailing_whitespace(markers.assistant_start, tokenizer)
    turn_end_ids = _tokenize_marker(markers.turn_end, tokenizer)

    # Initialize all labels as masked
    labels = [IGNORE_INDEX] * len(input_ids)

    # Find all assistant turns and mark them as trainable
    i = 0
    while i < len(input_ids):
        # Look for assistant start marker
        if assistant_start_ids:
            asst_start_pos = _find_subsequence(input_ids, assistant_start_ids, i)
        else:
            asst_start_pos = -1

        if asst_start_pos == -1:
            break  # No more assistant turns

        # Move past the assistant start marker (don't include marker in labels)
        content_start = asst_start_pos + len(assistant_start_ids)

        # Find the end of this assistant turn
        # Look for either turn_end or next user_start, whichever comes first
        turn_end_pos = len(input_ids)  # Default to end of sequence

        if turn_end_ids:
            end_pos = _find_subsequence(input_ids, turn_end_ids, content_start)
            if end_pos != -1:
                # Include the turn_end token in labels
                turn_end_pos = end_pos + len(turn_end_ids)

        if user_start_ids:
            next_user_pos = _find_subsequence(input_ids, user_start_ids, content_start)
            if next_user_pos != -1:
                turn_end_pos = min(turn_end_pos, next_user_pos)

        # Mark tokens from content_start to turn_end_pos as trainable
        for j in range(content_start, min(turn_end_pos, len(input_ids))):
            labels[j] = input_ids[j]

        # Move to search for next assistant turn
        i = turn_end_pos

    # Validate that we found trainable tokens
    trainable_count = sum(1 for label in labels if label != IGNORE_INDEX)
    if trainable_count == 0:
        logger.warning(
            "No trainable tokens found after label masking. "
            "This may cause NaN loss during training. "
            "Check that the chat template markers match your model's format."
        )

    return labels


def get_token_based_labels_with_ignore_empty_think(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizerBase,
    markers: ChatMarkerConfig,
    ignore_empty_think: bool = False,
) -> List[int]:
    """
    Create labels using token-based pattern matching, with support for ignore_empty_think.

    When ignore_empty_think is True, empty <think></think> patterns at the start of
    assistant responses are masked (set to IGNORE_INDEX) to prevent the model from
    learning to produce empty thinking blocks.
    """
    labels = get_token_based_labels(input_ids, tokenizer, markers)

    if not ignore_empty_think:
        return labels

    # Handle empty think tags - find and mask them
    # Tokenize the empty think pattern
    empty_think_pattern = "<think></think>"
    empty_think_ids = _tokenize_marker(empty_think_pattern, tokenizer)

    if not empty_think_ids:
        return labels

    # Find and mask empty think patterns that appear at start of assistant responses
    # IMPORTANT: Use the same tokenization as get_token_based_labels() to ensure
    # content_start positions match where tokens were actually marked as trainable
    assistant_start_ids = _tokenize_marker_without_trailing_whitespace(markers.assistant_start, tokenizer)
    user_start_ids = _tokenize_marker_without_trailing_whitespace(markers.user_start, tokenizer)
    turn_end_ids = _tokenize_marker(markers.turn_end, tokenizer)

    i = 0
    while i < len(input_ids):
        if assistant_start_ids:
            asst_start_pos = _find_subsequence(input_ids, assistant_start_ids, i)
        else:
            break

        if asst_start_pos == -1:
            break

        content_start = asst_start_pos + len(assistant_start_ids)

        # Check if empty think pattern is right after assistant start
        if input_ids[content_start : content_start + len(empty_think_ids)] == empty_think_ids:
            # Mask the empty think tokens
            for j in range(content_start, content_start + len(empty_think_ids)):
                if j < len(labels):
                    labels[j] = IGNORE_INDEX

        # Find the end of this assistant turn to skip past it entirely
        # This prevents false matches if the response contains marker patterns
        turn_end_pos = len(input_ids)  # Default to end of sequence

        if turn_end_ids:
            end_pos = _find_subsequence(input_ids, turn_end_ids, content_start)
            if end_pos != -1:
                turn_end_pos = end_pos + len(turn_end_ids)

        if user_start_ids:
            next_user_pos = _find_subsequence(input_ids, user_start_ids, content_start)
            if next_user_pos != -1:
                turn_end_pos = min(turn_end_pos, next_user_pos)

        # Move past the entire assistant turn
        i = turn_end_pos

    return labels
