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
Convert a CSV or Parquet file with SQL completion data to HuggingFace FIM format.

This script outputs datasets with prompt/response columns for length-based label masking:
- prompt: Full chat template + FIM tokens up to <|fim_middle|>
- response: Just the completion (OUTPUT)

The input file has columns: TYPE, INPUT, OUTPUT
- TYPE: The type of fillMe task (e.g., "fillMe_general_identifier") - ignored in output
- INPUT: A JSON array with text content containing context and partial query with <fillMe>
- OUTPUT: The value to replace <fillMe>

This is a standalone script with no external dependencies on verl/snow_rlhf.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict
from typing import Optional

import datasets
import pandas as pd

# =============================================================================
# Template Parsing Functions (inline, no external dependencies)
# =============================================================================


def load_template(template_path: str) -> str:
    """Load a template file."""
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_template_section(template_content: str, section_name: str) -> str:
    """
    Parse a Go-style template section from the template content.

    Extracts content between {{define "section_name"}} and {{end}}.
    """
    pattern = rf'\{{\{{define\s+"{section_name}"\}}\}}(.*?)\{{\{{end\}}\}}'
    match = re.search(pattern, template_content, re.DOTALL)

    if match:
        return match.group(1).strip()

    raise ValueError(f"Template section '{section_name}' not found in template")


def render_system_prompt(template_content: str) -> str:
    """Extract and return the system prompt from the template."""
    return parse_template_section(template_content, "system")


def render_user_prompt(
    template_content: str,
    sql_history: str = "",
    ddl_context: str = "",
    last_few_executed: str = "",
    file_content: str = "",
) -> str:
    """
    Render the user prompt section with context fields.
    """
    user_section = parse_template_section(template_content, "user")

    # Substitute placeholders
    user_section = user_section.replace("{{.SQL_HISTORY}}", sql_history)
    user_section = user_section.replace("{{.DDL_CONTEXT}}", ddl_context)
    user_section = user_section.replace("{{.LAST_FEW_EXECUTED}}", last_few_executed)
    user_section = user_section.replace("{{.FILE_CONTENT}}", file_content)

    return user_section


def render_assistant_prefix(template_content: str, prefix: str, suffix: str) -> str:
    """
    Render the assistant prefix (FIM structure up to and including <|fim_middle|>).

    This content is appended to the prompt and will be masked during training.
    """
    assistant_section = parse_template_section(template_content, "assistant")

    # Substitute placeholders
    assistant_section = assistant_section.replace("{{.PREFIX}}", prefix)
    assistant_section = assistant_section.replace("{{.SUFFIX}}", suffix)

    return assistant_section


# =============================================================================
# JSON Parsing Functions (standalone)
# =============================================================================


def _sanitize_json_input(input_text: str) -> str:
    """
    Sanitize JSON that has malformed content in text values.

    Handles:
    1. Unescaped double quotes in SQL
    2. Invalid escape sequences (e.g., \\d, \\s from regex patterns)
    """
    result = []
    i = 0
    text = input_text

    valid_escapes = set('"\\bfnrtu/')

    while i < len(text):
        text_marker = '"text": "'
        marker_pos = text.find(text_marker, i)

        if marker_pos == -1:
            result.append(text[i:])
            break

        result.append(text[i : marker_pos + len(text_marker)])
        content_start = marker_pos + len(text_marker)

        sql_history_end = text.find("</sql_history>", content_start)

        if sql_history_end != -1 and sql_history_end < len(text) - 100:
            transition = text.find('"}, {"type":', sql_history_end)
            if transition != -1:
                end_pos = transition
            else:
                end_pos = text.find('"}', sql_history_end)
        else:
            end_pos = text.rfind('"}]')

        if end_pos == -1 or end_pos <= content_start:
            result.append(text[content_start:])
            break

        j = content_start
        content_chars = []
        while j < end_pos:
            if text[j] == '"':
                content_chars.append('\\"')
            elif text[j] == "\\" and j + 1 < len(text):
                next_char = text[j + 1]
                if next_char in valid_escapes:
                    content_chars.append(text[j : j + 2])
                    j += 1
                else:
                    content_chars.append("\\\\")
            else:
                content_chars.append(text[j])
            j += 1

        result.append("".join(content_chars))
        i = end_pos

    return "".join(result)


def parse_input_json(input_text: str) -> Dict:
    """
    Parse the INPUT column which is a JSON array of text objects.

    Supports two input formats:

    New format (JSON context object):
        [{"type": "text", "text": '{"sql_history": "...", "last_few_executed": "...",
          "file_content": "...", "ddl_context": "..."}'},
         {"type": "text", "text": "SELECT * FROM <fillMe> WHERE ..."}]

    Old format (XML-style tags):
        [{"type": "text", "text": "<sql_history>...</sql_history>"},
         {"type": "text", "text": "...SQL query with <fillMe> placeholder..."}]

    Returns:
        dict: Contains prefix, suffix, sql_history, ddl_context, last_few_executed,
              file_content, used_sanitization, and parse_failed fields.
    """
    result = {
        "prefix": "",
        "suffix": "",
        "sql_history": "",
        "ddl_context": "",
        "last_few_executed": "",
        "file_content": "",
        "used_sanitization": False,
        "parse_failed": False,
    }

    if pd.isna(input_text) or input_text == "":
        return result

    input_array = None

    try:
        input_array = json.loads(input_text)
        if not isinstance(input_array, list):
            input_array = None
    except (json.JSONDecodeError, TypeError):
        try:
            sanitized = _sanitize_json_input(input_text)
            input_array = json.loads(sanitized)
            result["used_sanitization"] = True
            if not isinstance(input_array, list):
                input_array = None
        except (json.JSONDecodeError, TypeError):
            input_array = None

    if input_array is None:
        result["prefix"] = str(input_text)
        result["parse_failed"] = True
        return result

    fillme_text = ""
    found_context = False

    for item in input_array:
        if isinstance(item, dict) and item.get("type") == "text":
            text_content = item.get("text", "")

            # Try to parse as JSON context object (new format)
            try:
                context_obj = json.loads(text_content)
                if isinstance(context_obj, dict) and any(
                    k in context_obj for k in ["sql_history", "last_few_executed", "file_content", "ddl_context"]
                ):
                    result["sql_history"] = context_obj.get("sql_history") or ""
                    result["last_few_executed"] = context_obj.get("last_few_executed") or ""
                    result["file_content"] = context_obj.get("file_content") or ""
                    result["ddl_context"] = context_obj.get("ddl_context") or ""
                    found_context = True
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

            # Check for old format with <sql_history> tags
            if re.search(r"<sql_history>", text_content, re.IGNORECASE):
                result["sql_history"] = re.sub(r"</?sql_history>", "", text_content, flags=re.IGNORECASE).strip()
                found_context = True
            elif re.search(r"<fillme>", text_content, re.IGNORECASE):
                fillme_text = text_content
            else:
                if not found_context and not fillme_text:
                    result["sql_history"] = text_content
                    found_context = True
                elif not fillme_text:
                    fillme_text = text_content

    # Extract prefix and suffix from fillMe text
    if fillme_text:
        if re.search(r"<fillme>", fillme_text, re.IGNORECASE):
            parts = re.split(r"<fillme>", fillme_text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                result["prefix"] = parts[0]
                result["suffix"] = parts[1]
            else:
                result["prefix"] = re.sub(r"<fillme>", "", fillme_text, flags=re.IGNORECASE)
        else:
            result["prefix"] = fillme_text

    return result


# =============================================================================
# Main Conversion Functions
# =============================================================================


def build_fim_prompt(
    template_content: str,
    tokenizer,
    prefix: str,
    suffix: str,
    sql_history: str = "",
    ddl_context: str = "",
    last_few_executed: str = "",
    file_content: str = "",
) -> str:
    """
    Build the full FIM prompt including chat template and FIM structure.

    The prompt includes everything up to and including <|fim_middle|>.
    Everything in this prompt will be masked (-100) during training.

    Returns:
        str: The full prompt string
    """
    # Get template sections
    system_content = render_system_prompt(template_content)
    user_content = render_user_prompt(
        template_content,
        sql_history=sql_history,
        ddl_context=ddl_context,
        last_few_executed=last_few_executed,
        file_content=file_content,
    )

    # Build messages for chat template
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Apply chat template with generation prompt (adds "<|im_start|>assistant\n")
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Append FIM structure (this is part of prompt, so it will be masked)
    assistant_prefix = render_assistant_prefix(template_content, prefix, suffix)
    prompt = prompt + assistant_prefix

    return prompt


def read_input_file(file_path: str) -> pd.DataFrame:
    """Read input file, auto-detecting format based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        print("Detected CSV format")
        return pd.read_csv(file_path)
    elif ext in [".parquet", ".pq"]:
        print("Detected Parquet format")
        return pd.read_parquet(file_path)
    else:
        print(f"Unknown extension '{ext}', attempting to read as CSV")
        return pd.read_csv(file_path)


def convert_to_hf_fim_dataset(
    input_file_path: str,
    output_dir: str,
    tokenizer_name: str,
    template_path: Optional[str] = None,
    input_col: str = "INPUT",
    output_col: str = "OUTPUT",
    save_format: str = "disk",
    allow_empty_output: bool = True,
    allow_empty_prefix: bool = True,
    verbose: bool = False,
):
    """
    Convert CSV or Parquet file to HuggingFace FIM format.

    Output format: prompt/response columns for length-based label masking.
    - prompt: Full chat template + FIM tokens up to <|fim_middle|>
    - response: Just the completion (OUTPUT)

    Args:
        input_file_path: Path to the input CSV or Parquet file
        output_dir: Path to the output directory
        tokenizer_name: HuggingFace tokenizer name or path
        template_path: Path to FIM template file (optional, uses default)
        input_col: Name of the input column
        output_col: Name of the output column
        save_format: Output format - "disk" or "parquet"
        allow_empty_output: If True, allow empty OUTPUT (response will be empty string)
        allow_empty_prefix: If True, allow <fillMe> at start (no prefix)
        verbose: If True, show detailed debug info

    Returns:
        datasets.Dataset: The converted HuggingFace dataset
    """
    from transformers import AutoTokenizer

    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file '{input_file_path}' not found.")

    # Generate output path if not provided
    if output_dir is None:
        base_name = os.path.splitext(input_file_path)[0]
        output_dir = f"{base_name}_hf_fim_dataset"

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Load template
    if template_path is None:
        # Default template location: same directory as this script
        script_dir = Path(__file__).parent
        template_path = script_dir / "fim_prompt.tmpl"

    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file '{template_path}' not found.")

    print(f"Loading template: {template_path}")
    template_content = load_template(str(template_path))

    print(f"Converting {input_file_path} to HuggingFace FIM format")
    print(f"Output directory: {output_dir}")

    # Read the input file
    print("\nReading input file...")
    df = read_input_file(input_file_path)

    print(f"Loaded {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")

    # Find columns (case-insensitive)
    col_mapping = {}
    for col in df.columns:
        if col.upper() == input_col.upper():
            col_mapping["input"] = col
        elif col.upper() == output_col.upper():
            col_mapping["output"] = col

    if "input" not in col_mapping:
        raise ValueError(f"Input column '{input_col}' not found. Available: {list(df.columns)}")
    if "output" not in col_mapping:
        raise ValueError(f"Output column '{output_col}' not found. Available: {list(df.columns)}")

    print("\nUsing columns:")
    print(f"  Input: {col_mapping['input']}")
    print(f"  Output: {col_mapping['output']}")

    # Convert to FIM format
    print("\nConverting to HuggingFace FIM format...")
    print(f"  Options: allow_empty_output={allow_empty_output}, allow_empty_prefix={allow_empty_prefix}")

    hf_data = []
    skipped_counts = {
        "empty_input": 0,
        "empty_output": 0,
        "empty_prefix": 0,
        "json_parse_error": 0,
    }
    sanitization_count = 0

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing row {idx:,}/{len(df):,}", end="\r")

        # Check for null/empty INPUT
        input_value = row[col_mapping["input"]]
        if pd.isna(input_value) or input_value == "" or input_value is None:
            skipped_counts["empty_input"] += 1
            continue

        # Check for null/empty OUTPUT
        output_value = row[col_mapping["output"]]
        output_is_empty = pd.isna(output_value) or output_value == "" or output_value is None

        if output_is_empty:
            if allow_empty_output:
                response = ""  # Empty response is allowed
            else:
                skipped_counts["empty_output"] += 1
                continue
        else:
            response = str(output_value)

        # Parse the input JSON
        parsed = parse_input_json(input_value)

        if parsed["used_sanitization"]:
            sanitization_count += 1

        if parsed["parse_failed"]:
            skipped_counts["json_parse_error"] += 1
            continue

        # Check if prefix is empty
        prefix_is_empty = not parsed["prefix"] or parsed["prefix"].strip() == ""

        if prefix_is_empty and not allow_empty_prefix:
            skipped_counts["empty_prefix"] += 1
            continue

        # Build the FIM prompt
        prompt = build_fim_prompt(
            template_content=template_content,
            tokenizer=tokenizer,
            prefix=parsed["prefix"],
            suffix=parsed["suffix"],
            sql_history=parsed["sql_history"],
            ddl_context=parsed["ddl_context"],
            last_few_executed=parsed["last_few_executed"],
            file_content=parsed["file_content"],
        )

        # Create the data item with prompt/response columns
        data_item = {
            "prompt": prompt,
            "response": response,
        }

        hf_data.append(data_item)

    # Print statistics
    if sanitization_count > 0:
        print(f"\n  Sanitized {sanitization_count:,} rows (fixed JSON issues)")

    total_skipped = sum(skipped_counts.values())
    if total_skipped > 0:
        print(f"\n  Skipped {total_skipped:,} rows:")
        for reason, count in skipped_counts.items():
            if count > 0:
                print(f"    - {reason}: {count:,}")

    print(f"\nConverted {len(hf_data):,} rows (from {len(df):,} total).              ")

    if len(hf_data) == 0:
        raise ValueError("No valid rows found after filtering.")

    # Create HuggingFace dataset
    print("\nCreating HuggingFace dataset...")
    hf_dataset = datasets.Dataset.from_list(hf_data)

    print(f"Dataset features: {hf_dataset.features}")
    print(f"Dataset size: {len(hf_dataset):,} examples")

    # Show sample
    print("\nSample entry:")
    sample = hf_dataset[0]
    prompt_preview = sample["prompt"]
    print(f"  Prompt: {prompt_preview}")
    print(f"  Response: {sample['response']}")
    print(f"  Complete_Message: {sample}")

    # Save the dataset
    print(f"\nSaving dataset to: {output_dir}")
    if save_format == "disk":
        hf_dataset.save_to_disk(output_dir)
        print("Saved as HuggingFace dataset directory")
    elif save_format == "parquet":
        os.makedirs(output_dir, exist_ok=True)
        parquet_path = os.path.join(output_dir, "data.parquet")
        hf_dataset.to_parquet(parquet_path)
        print(f"Saved as parquet file: {parquet_path}")
    else:
        raise ValueError(f"Unknown save_format: {save_format}. Use 'disk' or 'parquet'")

    print("\nConversion complete!")
    return hf_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert SQL completion CSV/Parquet to HuggingFace FIM format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python csv_to_hf_fim.py input.parquet -o output_hf --tokenizer Qwen/Qwen3-4B-Instruct-2507

  # With custom template
  python csv_to_hf_fim.py input.csv -o output_hf --tokenizer Qwen/Qwen3-4B-Instruct-2507 --template my_template.tmpl

  # Output as parquet
  python csv_to_hf_fim.py input.parquet -o output_hf --tokenizer Qwen/Qwen3-4B-Instruct-2507 --format parquet
""",
    )
    parser.add_argument("input_file", help="Path to the input CSV or Parquet file")
    parser.add_argument("-o", "--output", help="Output directory path (default: <input_basename>_hf_fim_dataset)")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace tokenizer name or path (required for chat template)",
    )
    parser.add_argument(
        "--template",
        help="Path to FIM template file (default: fim_prompt.tmpl in same directory)",
    )
    parser.add_argument("--input-col", default="INPUT", help="Name of the input column (default: INPUT)")
    parser.add_argument("--output-col", default="OUTPUT", help="Name of the output column (default: OUTPUT)")
    parser.add_argument(
        "--format",
        choices=["disk", "parquet"],
        default="disk",
        help="Output format: 'disk' for HF save_to_disk, 'parquet' for parquet file (default: disk)",
    )
    parser.add_argument(
        "--skip-empty-output",
        action="store_true",
        help="Skip rows with null/empty OUTPUT (default: include them with empty response)",
    )
    parser.add_argument(
        "--skip-empty-prefix",
        action="store_true",
        help="Skip rows where <fillMe> is at the start (empty prefix). Default: include them.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed debug info for skipped rows",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CSV/Parquet to HuggingFace FIM Format Converter")
    print("=" * 80)

    try:
        convert_to_hf_fim_dataset(
            input_file_path=args.input_file,
            output_dir=args.output,
            tokenizer_name=args.tokenizer,
            template_path=args.template,
            input_col=args.input_col,
            output_col=args.output_col,
            save_format=args.format,
            allow_empty_output=not args.skip_empty_output,
            allow_empty_prefix=not args.skip_empty_prefix,
            verbose=args.verbose,
        )
        print("\n" + "=" * 80)
        print("SUCCESS")
        print("=" * 80)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
