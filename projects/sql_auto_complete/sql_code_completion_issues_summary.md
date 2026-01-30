# Arctic Training Framework: SQL Code Completion Issues & Solutions

**Project:** SQL Auto-Complete Fine-Tuning  
**Date:** January 2026  
**Status:** Resolved

---

## Executive Summary

During SQL code completion model training using the Arctic Training framework, we identified several issues causing performance degradation compared to baseline models trained with other frameworks. This document summarizes the root causes and the implemented solutions, presented in the order they were addressed.

---

## Issue 1: Prompt/Response Format with Assistant Message Masking

### Problem Statement
The framework originally only supported the `messages` format (chat-template style) for instruction tuning. For SQL code completion, we needed to support a simpler `prompt`/`response` column format where training focuses only on the assistant's response tokens (the completion), while masking the prompt tokens from loss computation.

Additionally, when using instruct-tuned models like Qwen-Instruct, the framework needed to use the correct end-of-turn tokens (`<|im_end|>` for Qwen, `<|eot_id|>` for Llama-3) rather than the generic EOS token to ensure proper model behavior during inference.

### Approach
Extended the data factory to auto-detect and process `prompt`/`response` format datasets:
- Added `tokenize_prompt_response()` method that correctly masks prompt tokens (sets labels to -100) and trains only on response tokens
- Implemented model-aware end-of-turn token selection that detects chat-template tokens from the tokenizer vocabulary
- For Qwen-Instruct: uses `<|im_end|>`; for Llama-3-Instruct: uses `<|eot_id|>`; falls back to standard EOS for base models

**Key commits:** `4b62b42`, `7dfdeb0`, `bcce04a`, `4fd5367`

---

## Issue 2: NaN/Division-by-Zero in Loss Computation

### Problem Statement
When batches contained zero trainable tokens (all labels masked to -100), the loss computation resulted in NaN or division-by-zero errors, crashing training. This occurred with certain data distributions where packed batches might contain only prompt tokens.

### Approach
Added explicit handling for zero-token batches: return a zero loss that maintains gradient connectivity (using `logits.sum() * 0.0`) with appropriate warnings. Also added NaN/Inf checks after tiled loss computation for robustness.

**Key commits:** `cc5de11`, `69aa384`, `e264eec`, `145de6a`

---

## Issue 3: Loss Dilution with Sample Packing

### Problem Statement
When using sample packing (concatenating multiple training samples into a single sequence for GPU efficiency), short-output samples were effectively ignored during training. For example, in a packed sequence with one 5-token output and one 1000-token output, the short sample contributed only 0.5% of the gradient signal—despite potentially being a harder reasoning task.

This "loss dilution" or "gradient starvation" caused the model to under-learn on short SQL completions (e.g., single table names, column references) while over-fitting to longer completions.

### Approach
Implemented **sample-weighted loss computation** (`sample_weighted_loss: true`). Instead of averaging loss over all tokens globally, we compute loss per-sample and average the per-sample losses. This ensures each training sample contributes equally regardless of response length.

**Key commits:** `803a1da`, `91a2b92`

---

## Issue 4: Sample-Weighted Loss for Liger Models

### Problem Statement
The Liger kernel provides memory-efficient fused cross-entropy loss computation, but it doesn't return logits—only the final loss. This prevented us from computing per-sample losses when `model.type: liger` was configured. Users had to choose between memory efficiency (Liger) and loss balancing (sample-weighted loss).

### Approach
Modified the loss computation path to call Liger models **without labels** when `sample_weighted_loss` is enabled. This bypasses the fused loss kernel and returns logits, enabling manual sample-weighted loss computation. A warning is logged about the memory trade-off.

**Key commits:** `803a1da`, `4fd5367`

---

## Issue 5: Attention Isolation in Packed Samples

### Problem Statement
Testing revealed that position ID resets alone do **not** create attention isolation in HuggingFace models. Tokens from Sample B could attend to tokens from Sample A within a packed sequence, causing "cross-contamination" that degrades model quality.

### Approach
Created comprehensive test suite (`test_attention_isolation.py`) to verify isolation behavior. Documented the limitation and identified that explicit 4D block-diagonal attention masks or `flash_attn_varlen_func` with `cu_seqlens` would be required for true isolation. For current use, length grouping and sample-weighted loss mitigate the impact.

**Key commits:** `ce38037`, `91a2b92`

---

## Issue 6: Liger Kernel with Sequence Parallelism and Evaluation

### Problem Statement
When using Liger models with sequence parallelism (SP > 1) and evaluation enabled, the fused cross-entropy returned `None` during evaluation, crashing training.

### Approach
Modified the loss path to detect evaluation mode (via `model_unwrapped.training`) and fall back to tiled logits+loss computation for evaluation, while still using Liger's efficient fused loss for training steps.

**Key commits:** Referenced in `sft_trainer.py` updates

---

## Summary of Changes

| Area | Files Modified | Key Improvement |
|------|----------------|-----------------|
| Sample-Weighted Loss | `sft_trainer.py` | Equal gradient contribution per sample |
| EOS Token Handling | `sft_factory.py` | Correct chat-template tokens |
| Zero-Token Handling | `sft_trainer.py` | Graceful handling with gradient connectivity |
| Testing | `test_attention_isolation.py`, `test_fim_label_masking.py` | Comprehensive validation |
| Documentation | `fim_training.md` | Usage guidance |

---

## Configuration for SQL Code Completion

```yaml
data:
  pack_samples: true
  sample_weighted_loss: true  # Critical for short-output samples
  
model:
  type: liger  # Memory efficient, now compatible with sample_weighted_loss
```

---

## Verification

All fixes validated via:
1. Unit tests for loss masking and tokenization
2. Attention isolation tests (gradient leak detection)
3. Sample-weighted loss validation script confirming per-sample averaging
4. End-to-end training runs showing improved performance on short completions
