# SQL Auto-Complete Training

This project contains tools for training SQL auto-completion models using FIM (Fill-in-the-Middle) format.

## Overview

The SQL auto-complete model learns to complete SQL queries given:
- **Prefix**: SQL text before the cursor
- **Suffix**: SQL text after the cursor
- **Context**: SQL history, DDL context, recent queries, file content

The training uses a **length-based label masking** approach where:
- The `prompt` column contains everything up to `<|fim_middle|>` (masked during training)
- The `response` column contains only the completion (trained on)

## Files

| File | Description |
|------|-------------|
| `csv_to_hf_fim.py` | Convert CSV/Parquet data to HuggingFace FIM format |
| `fim_prompt.tmpl` | Template for FIM prompts (system, user, assistant structure) |

## Quick Start

### 1. Prepare Training Data

Convert your CSV/Parquet data to HuggingFace FIM format:

```bash
python csv_to_hf_fim.py \
    /path/to/input.parquet \
    -o /path/to/output_hf \
    --tokenizer Qwen/Qwen3-4B-Instruct-2507
```

### 2. Configure Training

Create a training config that points to your generated dataset:

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

## Data Format

### Input Format (CSV/Parquet)

| Column | Description |
|--------|-------------|
| INPUT | JSON with context and `<fillMe>` placeholder |
| OUTPUT | The completion to insert at `<fillMe>` |

**Example INPUT:**
```json
[
  {"type": "text", "text": "{\"sql_history\": \"SELECT * FROM users;\", \"ddl_context\": \"CREATE TABLE orders...\"}"},
  {"type": "text", "text": "SELECT * FROM <fillMe> WHERE id = 1"}
]
```

**Example OUTPUT:**
```
orders
```

### Output Format (HuggingFace)

| Column | Description |
|--------|-------------|
| prompt | Full prompt with FIM tokens up to `<\|fim_middle\|>` |
| response | Just the completion (OUTPUT) |

## Script Options

```bash
python csv_to_hf_fim.py --help
```

| Option | Description |
|--------|-------------|
| `input_file` | Path to input CSV/Parquet |
| `-o, --output` | Output directory (default: `<input>_hf_fim_dataset`) |
| `--tokenizer` | **Required**. HuggingFace tokenizer name/path |
| `--template` | Path to FIM template (default: `fim_prompt.tmpl`) |
| `--format` | Output format: `disk` or `parquet` |
| `--skip-empty-output` | Skip rows with empty OUTPUT |
| `--skip-empty-prefix` | Skip rows with `<fillMe>` at start |

## Examples

```bash
# Basic usage with Qwen tokenizer
python csv_to_hf_fim.py data.parquet \
    -o fim_dataset \
    --tokenizer Qwen/Qwen3-4B-Instruct-2507

# Save as parquet file
python csv_to_hf_fim.py data.csv \
    -o fim_dataset \
    --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
    --format parquet

# Skip empty outputs
python csv_to_hf_fim.py data.parquet \
    -o fim_dataset \
    --tokenizer Qwen/Qwen3-4B-Instruct-2507 \
    --skip-empty-output
```

## Template Structure

The `fim_prompt.tmpl` file defines three sections:

1. **system**: Instructions for the SQL completion model
2. **user**: Context fields (SQL history, DDL, recent queries, file content)
3. **assistant**: FIM structure with `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`

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

## Troubleshooting

### NaN Loss

The trainer includes protection against NaN loss. If you see warnings:
1. Check your data has non-empty OUTPUT values
2. Verify the tokenizer matches the model
3. Check for data distribution issues with packing

### Empty Outputs

Empty outputs are **allowed by default**. The model learns when "no completion needed".
Use `--skip-empty-output` during data generation to exclude them.

## Related Files

| File | Location | Purpose |
|------|----------|---------|
| `sft_factory.py` | `arctic_training/data/` | Tokenization and label masking |
| `hf_source.py` | `arctic_training/data/` | Data source classes |
| `sft_trainer.py` | `arctic_training/trainer/` | SFT trainer with NaN handling |
| `fim_training.md` | `docs/` | Full training documentation |
| `test_fim_label_masking.py` | `tests/data/` | Tests for FIM label masking |
