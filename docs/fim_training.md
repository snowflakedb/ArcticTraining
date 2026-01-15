# FIM (Fill-in-the-Middle) Training Guide

This guide covers training models for SQL auto-completion using FIM format.

## Overview

FIM training teaches models to complete code given prefix and suffix context:
- **Prefix**: Code before the cursor
- **Suffix**: Code after the cursor
- **Output**: What should be inserted at the cursor

The training uses a **length-based label masking** approach where:
- The `prompt` column contains everything up to `<|fim_middle|>` (masked during training)
- The `response` column contains only the completion (trained on)

This is simpler and more robust than token-pattern matching approaches.

## Quick Start

### 1. Prepare Training Data

Convert your CSV/Parquet data to HuggingFace FIM format:

```bash
python projects/sql_auto_complete/csv_to_hf_fim.py \
    /path/to/input.parquet \
    -o /path/to/output_hf \
    --tokenizer Qwen/Qwen3-4B-Instruct-2507
```

**Input format** (CSV/Parquet columns):

| Column | Description |
|--------|-------------|
| INPUT | JSON with context and `<fillMe>` placeholder |
| OUTPUT | The completion to insert at `<fillMe>` |

**Output format** (HuggingFace dataset):

| Column | Description |
|--------|-------------|
| prompt | Full prompt with FIM tokens up to `<\|fim_middle\|>` |
| response | Just the completion (OUTPUT) |

### 2. Configure Training

Create your training config (e.g., `config.yml`):

```yaml
type: sft
model:
  type: liger
  name_or_path: Qwen/Qwen3-4B-Instruct-2507

data:
  sources:
    - name_or_path: /path/to/output_hf
      # type is auto-detected from columns
  max_length: 262144
  pack_samples: true

# ... optimizer, scheduler, etc.
```

### 3. Run Training

```bash
python -m arctic_training config.yml
```

## How Loss Masking Works

The trainer **automatically detects** the dataset format based on columns:

| Dataset Columns | Masking Approach |
|-----------------|------------------|
| `prompt` + `response` | Length-based: prompt masked, response trained |
| `messages` | Character-offset based (legacy) |

For FIM format:
- Everything in `prompt` column → masked (-100)
- Everything in `response` column → trained
- `<|fim_middle|>` is the last token in prompt

### Why Length-Based Masking?

The key insight is that by splitting prompt and response **before tokenization**, the loss boundary is deterministic:

```python
# Tokenize separately
prompt_ids = tokenizer(prompt)["input_ids"]
response_ids = tokenizer(response + eos)["input_ids"]

# Labels: mask prompt, train response
labels = [-100] * len(prompt_ids) + response_ids
```

This avoids:
- Character-to-token mapping edge cases
- Token pattern matching complexity
- Model-family-specific marker detection

## Data Generation Script

### `csv_to_hf_fim.py` Options

```bash
python projects/sql_auto_complete/csv_to_hf_fim.py --help
```

| Option | Description |
|--------|-------------|
| `input_file` | Path to input CSV/Parquet |
| `-o, --output` | Output directory (default: `<input>_hf_fim_dataset`) |
| `--tokenizer` | **Required**. HuggingFace tokenizer name/path |
| `--template` | Path to FIM template (default: `templates/fim_prompt.tmpl`) |
| `--format` | Output format: `disk` or `parquet` |
| `--skip-empty-output` | Skip rows with empty OUTPUT |
| `--skip-empty-prefix` | Skip rows with `<fillMe>` at start |

### Examples

```bash
# Basic usage with Qwen tokenizer
python arctic_training/csv_to_hf_fim.py data.parquet \
    -o fim_dataset \
    --tokenizer Qwen/Qwen3-4B-Instruct-2507

# Save as parquet file
python arctic_training/csv_to_hf_fim.py data.csv \
    -o fim_dataset \
    --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
    --format parquet

# Skip empty outputs
python arctic_training/csv_to_hf_fim.py data.parquet \
    -o fim_dataset \
    --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
    --skip-empty-output
```

## Template Reference

The FIM prompt template is at `projects/sql_auto_complete/fim_prompt.tmpl`:

```
{{define "system"}}You are a Snowflake SQL code-completion model...{{end}}

{{define "user"}}Here is the full context...
# SQL History: {{.SQL_HISTORY}}
# DDL Context: {{.DDL_CONTEXT}}
# Last Few Executed: {{.LAST_FEW_EXECUTED}}
# File Content: {{.FILE_CONTENT}}
{{end}}

{{define "assistant"}}<|fim_prefix|>{{.PREFIX}}<|fim_suffix|>{{.SUFFIX}}<|fim_middle|>{{end}}
```

The generated prompt structure:
```
<|im_start|>system
{system_content}
<|im_end|>
<|im_start|>user
{user_content with context}
<|im_end|>
<|im_start|>assistant
<|fim_prefix|>{PREFIX}<|fim_suffix|>{SUFFIX}<|fim_middle|>
```

Everything up to and including `<|fim_middle|>` is in the `prompt` column (masked).
The `response` column contains only the completion OUTPUT.

## Config Options

### Data Config

```yaml
data:
  type: sft
  sources:
    - name_or_path: /path/to/dataset
      type: huggingface_prompt_response  # Optional, auto-detected

  max_length: 262144      # Maximum sequence length
  pack_samples: true      # Pack multiple samples into one
  mask_inputs: true       # Mask non-assistant tokens (for messages format)
  filter_samples: true    # Filter samples exceeding max_length
```

### Data Source Types

| Type | Columns | Use Case |
|------|---------|----------|
| `huggingface_prompt_response` | `prompt`, `response` | FIM training |
| `huggingface_instruct` | `messages` | Standard SFT |
| `huggingface` | varies | General datasets |

The type is auto-detected from columns, so you usually don't need to specify it.

## Troubleshooting

### NaN Loss

The trainer includes protection against NaN loss from all-masked batches:

```
WARNING: Batch has no trainable tokens across all SP ranks (all labels are -100).
Returning zero loss. This may indicate data issues...
```

If you see this warning frequently:
1. Check your data has non-empty OUTPUT values
2. Verify the tokenizer matches the model
3. Check for data distribution issues with packing

### Empty Outputs

Empty outputs are **allowed by default**. The model learns when "no completion needed".
If you want to skip them, use `--skip-empty-output` during data generation.

### Token Limits

If samples are being filtered out:
1. Check `max_length` in your config
2. Use `filter_samples: false` to see warnings instead of filtering
3. Consider using `pack_samples: true` to efficiently use long sequences

## Files Reference

| File | Purpose |
|------|---------|
| `projects/sql_auto_complete/csv_to_hf_fim.py` | Convert data to FIM format |
| `projects/sql_auto_complete/fim_prompt.tmpl` | FIM prompt template |
| `arctic_training/data/sft_factory.py` | Tokenization and label masking |
| `arctic_training/data/hf_source.py` | Data source classes |
| `arctic_training/trainer/sft_trainer.py` | SFT trainer with NaN handling |
| `tests/data/test_fim_label_masking.py` | Tests for FIM label masking |

## Backwards Compatibility

Existing workflows using the `messages` format are **unchanged**:

```yaml
data:
  sources:
    - name_or_path: HuggingFaceH4/ultrachat_200k
  # Uses messages format, existing behavior preserved
```

The trainer auto-detects the format and applies the appropriate masking strategy.
