#!/usr/bin/env python3
"""
Convert a CSV or Parquet file with SQL completion data to HuggingFace dataset format.

The input file has columns: TYPE, INPUT, OUTPUT
- TYPE: The type of fillMe task (e.g., "fillMe_general_identifier") - ignored in output
- INPUT: A JSON array with text content containing sql_history and partial query with <fillMe>
- OUTPUT: The value to replace <fillMe>

The output HuggingFace dataset will have a "messages" column with user/assistant roles,
where the user message is rendered using the autocomplete FIM template.
"""

import argparse
import json
import os
import re
import sys

import datasets
import pandas as pd

# Special token to represent "no suggestion" for autocomplete.
# IMPORTANT: Empty strings ("") cause training issues:
#   - Empty strings create zero-width character ranges in conversation_text
#   - No tokens can overlap with a zero-width range
#   - This causes ALL labels to be masked (-100), resulting in NaN loss
#
# Options:
#   1. "" (empty string) - PROBLEMATIC: causes NaN loss in training
#   2. "<no_suggestion>" - Model learns to output this specific token
#   3. Skip these samples entirely (set allow_empty_output=False)
#
# Recommendation: Use a meaningful token the model can learn to predict
NO_SUGGESTION_TOKEN = "<no_suggestion>"

from snow_rlhf.verl_comp.utils.dataset.arctic_text_to_sql_r1_prompt import (
    render_autocomplete_assistant_response,
    render_autocomplete_system_prompt,
    render_autocomplete_user_context,
)


def _sanitize_json_input(input_text):
    """
    Sanitize JSON that has malformed content in text values.

    Handles two categories of JSON parsing failures:
    1. Unescaped double quotes in SQL (e.g., "gap" dates, "IT"."GITHUB" identifiers)
    2. Invalid escape sequences (e.g., \\d, \\s from regex patterns, \\' from SQL)

    The expected structure is:
    [{"type": "text", "text": "<sql_history>...</sql_history>"},
     {"type": "text", "text": "...<fillMe>..."}]

    Args:
        input_text (str): The raw INPUT string that failed JSON parsing

    Returns:
        str: Sanitized JSON string that should parse correctly
    """
    result = []
    i = 0
    text = input_text

    # Valid JSON escape characters after backslash
    valid_escapes = set('"\\bfnrtu/')

    while i < len(text):
        text_marker = '"text": "'
        marker_pos = text.find(text_marker, i)

        if marker_pos == -1:
            result.append(text[i:])
            break

        result.append(text[i : marker_pos + len(text_marker)])
        content_start = marker_pos + len(text_marker)

        # Determine where this text value ends
        # First text object contains </sql_history>, ends at "}, {"type":
        # Second text object contains <fillMe>, ends at "}] at the very end

        sql_history_end = text.find("</sql_history>", content_start)

        if sql_history_end != -1 and sql_history_end < len(text) - 100:
            # First text object - find the transition pattern after </sql_history>
            transition = text.find('"}, {"type":', sql_history_end)
            if transition != -1:
                end_pos = transition
            else:
                # Fallback - find "} near sql_history_end
                end_pos = text.find('"}', sql_history_end)
        else:
            # Second text object - ends at "}] at the very end
            end_pos = text.rfind('"}]')

        if end_pos == -1 or end_pos <= content_start:
            result.append(text[content_start:])
            break

        # Process content: escape unquoted quotes and invalid escape sequences
        j = content_start
        content_chars = []
        while j < end_pos:
            if text[j] == '"':
                # Escape unescaped double quotes
                content_chars.append('\\"')
            elif text[j] == "\\" and j + 1 < len(text):
                next_char = text[j + 1]
                if next_char in valid_escapes:
                    # Valid escape sequence - keep as is
                    content_chars.append(text[j : j + 2])
                    j += 1
                else:
                    # Invalid escape (e.g., \d, \s, \') - double the backslash
                    content_chars.append("\\\\")
            else:
                content_chars.append(text[j])
            j += 1

        result.append("".join(content_chars))
        i = end_pos

    return "".join(result)


def parse_input_json(input_text):
    """
    Parse the INPUT column which is a JSON array of text objects.

    Extracts:
    - sql_history: Content from <sql_history>...</sql_history> tags
    - prefix: Text before <fillMe> in the partial query
    - suffix: Text after <fillMe> in the partial query

    If JSON parsing fails (e.g., due to unescaped quotes or invalid escapes in SQL),
    sanitizes the JSON and retries parsing.

    Args:
        input_text (str): JSON string representing the input

    Returns:
        tuple: (prefix, suffix, sql_history, used_sanitization)
            - used_sanitization is True if JSON was sanitized before successful parsing
            - Returns (raw_input, "", "", False) if parsing fails even after sanitization
    """
    if pd.isna(input_text) or input_text == "":
        return "", "", "", False

    input_array = None
    used_sanitization = False

    try:
        # Try parsing the JSON array directly
        input_array = json.loads(input_text)

        # Validate that we got a list - json.loads can return primitives (number, bool, null)
        if not isinstance(input_array, list):
            print(f"Warning: JSON is not an array, got {type(input_array).__name__}")
            input_array = None

    except (json.JSONDecodeError, TypeError):
        # JSON parsing failed - try sanitizing and re-parsing
        try:
            sanitized = _sanitize_json_input(input_text)
            input_array = json.loads(sanitized)
            used_sanitization = True

            if not isinstance(input_array, list):
                print(f"Warning: Sanitized JSON is not an array, got {type(input_array).__name__}")
                input_array = None
        except (json.JSONDecodeError, TypeError):
            # Both original and sanitized parsing failed
            input_array = None

    # If parsing completely failed, return the raw input as fallback marker
    if input_array is None:
        return str(input_text), "", "", False

    # Extract context (sql_history) and fillMe text from separate objects
    sql_history = ""
    fillme_text = ""

    for item in input_array:
        if isinstance(item, dict) and item.get("type") == "text":
            text_content = item.get("text", "")

            # Check if this contains <sql_history> (case insensitive) - this is the SQL history context
            if re.search(r"<sql_history>", text_content, re.IGNORECASE):
                # Strip both opening and closing tags (case insensitive) to get the raw content
                sql_history = re.sub(r"</?sql_history>", "", text_content, flags=re.IGNORECASE).strip()
            # Check if this contains <fillMe> (case insensitive) - this is the completion point
            elif re.search(r"<fillme>", text_content, re.IGNORECASE):
                fillme_text = text_content
            # If neither <sql_history> nor <fillMe>, check if it's the first item (context) or second item (query)
            else:
                # If we haven't found sql_history yet and this is the first item, treat it as context
                if not sql_history and not fillme_text:
                    sql_history = text_content
                # If we have sql_history but no fillMe text, treat this as the query
                elif sql_history and not fillme_text:
                    fillme_text = text_content

    # Extract prefix and suffix from fillMe text
    prefix = ""
    suffix = ""

    if fillme_text:
        if re.search(r"<fillme>", fillme_text, re.IGNORECASE):
            # Split the text around <fillMe> (case insensitive) to get prefix and suffix
            parts = re.split(r"<fillme>", fillme_text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                prefix = parts[0]
                suffix = parts[1]
            else:
                # Fallback if multiple <fillMe> or other issues
                prefix = re.sub(r"<fillme>", "", fillme_text, flags=re.IGNORECASE)
                suffix = ""
        else:
            # If no <fillMe> found, treat the whole text as prefix
            prefix = fillme_text
            suffix = ""

    # Return with the sanitization flag
    return prefix, suffix, sql_history, used_sanitization


def create_messages(prefix, suffix, sql_history, output_content):
    """
    Create the messages list for HuggingFace instruct format with FIM in assistant response.

    Message structure:
    - System: Static behavior rules
    - User: Context only (sql_history, ddl_context, etc.)
    - Assistant: FIM tokens with prefix/suffix + the completion output

    This format trains the model to output: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{output}

    Args:
        prefix (str): Text before <fillMe> in the partial query
        suffix (str): Text after <fillMe> in the partial query
        sql_history (str): SQL history context
        output_content (str): The assistant output (the <fillMe> replacement)

    Returns:
        list: List of message dictionaries with role and content (system, user, assistant)
    """
    # Render the static system prompt (no context, just behavior rules)
    system_content = render_autocomplete_system_prompt()

    # Render the user message with ONLY context (no FIM tokens)
    user_content = render_autocomplete_user_context(
        sql_history=sql_history,
        ddl_context="",  # Empty as per plan
        last_few_executed_context="",  # Empty as per plan
        file_content="",  # Empty as per plan
    )

    # Render the assistant response with FIM tokens + output
    # Format: <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>{output}
    assistant_content = render_autocomplete_assistant_response(
        prefix=prefix,
        suffix=suffix,
        output=output_content,
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return messages


def read_input_file(file_path):
    """
    Read input file, auto-detecting format based on file extension.

    Args:
        file_path (str): Path to CSV or Parquet file

    Returns:
        pd.DataFrame: Loaded data
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        print("Detected CSV format")
        return pd.read_csv(file_path)
    elif ext in [".parquet", ".pq"]:
        print("Detected Parquet format")
        return pd.read_parquet(file_path)
    else:
        # Default to CSV for unknown extensions
        print(f"Unknown extension '{ext}', attempting to read as CSV")
        return pd.read_csv(file_path)


def convert_to_hf_dataset(
    input_file_path,
    output_dir=None,
    input_col="INPUT",
    output_col="OUTPUT",
    save_format="disk",  # "disk" for save_to_disk, "parquet" for to_parquet
    allow_empty_output=True,  # If True, use NO_SUGGESTION_TOKEN for null outputs instead of skipping
    allow_empty_prefix=True,  # If True, allow <fillMe> at start (no_context_keyword_only type)
    verbose=False,  # If True, show detailed debug info for skipped rows
):
    """
    Convert CSV or Parquet file to HuggingFace dataset format.

    Args:
        input_file_path (str): Path to the input CSV or Parquet file
        output_dir (str): Path to the output directory (optional, defaults to same dir as input)
        input_col (str): Name of the input column
        output_col (str): Name of the output column
        save_format (str): Output format - "disk" for HF save_to_disk, "parquet" for parquet file
        allow_empty_output (bool): If True, use NO_SUGGESTION_TOKEN for null outputs instead of skipping
        allow_empty_prefix (bool): If True, allow <fillMe> at start (no_context cases)
        verbose (bool): If True, show detailed debug info for skipped rows

    Returns:
        datasets.Dataset: The converted HuggingFace dataset
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Input file '{input_file_path}' not found.")

    # Generate output path if not provided
    if output_dir is None:
        base_name = os.path.splitext(input_file_path)[0]
        output_dir = f"{base_name}_hf_dataset"

    print(f"Converting {input_file_path} to HuggingFace dataset format")
    print(f"Output directory: {output_dir}")

    # Read the input file (CSV or Parquet)
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

    # Convert to HuggingFace format
    print("\nConverting to HuggingFace format...")
    print(f"  Options: allow_empty_output={allow_empty_output}, allow_empty_prefix={allow_empty_prefix}")
    hf_data = []
    skipped_counts = {
        "empty_input": 0,
        "empty_output": 0,
        "empty_prefix": 0,
        "json_parse_error": 0,
    }
    # Track rows where JSON sanitization was needed (informational, not errors)
    sanitization_count = 0
    debug_samples = {
        "empty_output": [],
        "empty_prefix": [],
        "json_parse_error": [],
    }

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processing row {idx:,}/{len(df):,}", end="\r")

        # Check for null/empty INPUT - skip if invalid
        input_value = row[col_mapping["input"]]
        if pd.isna(input_value) or input_value == "" or input_value is None:
            skipped_counts["empty_input"] += 1
            continue

        # Check for null/empty OUTPUT
        output_value = row[col_mapping["output"]]
        output_is_empty = pd.isna(output_value) or output_value == "" or output_value is None

        if output_is_empty:
            if allow_empty_output:
                # Use special token instead of empty string
                # Empty strings cause zero-width ranges → all labels masked → NaN loss
                output_content = NO_SUGGESTION_TOKEN
            else:
                skipped_counts["empty_output"] += 1
                if verbose and len(debug_samples["empty_output"]) < 3:
                    debug_samples["empty_output"].append((idx, str(input_value)[:100]))
                continue
        else:
            output_content = str(output_value)

        # Parse the input JSON to extract prefix, suffix, and sql_history
        prefix, suffix, sql_history, used_sanitization = parse_input_json(input_value)

        # Track rows where JSON sanitization was needed
        if used_sanitization:
            sanitization_count += 1

        # Check if parsing completely failed (both JSON and regex)
        # This happens when prefix = raw input and no other content extracted
        if prefix == str(input_value) and not suffix and not sql_history:
            skipped_counts["json_parse_error"] += 1
            if verbose and len(debug_samples["json_parse_error"]) < 3:
                debug_samples["json_parse_error"].append((idx, str(input_value)[:100]))
            continue

        # Check if prefix is empty (<fillMe> at the beginning)
        prefix_is_empty = not prefix or prefix.strip() == ""

        if prefix_is_empty:
            if allow_empty_prefix:
                # Valid case: no_context_keyword_only type where <fillMe> is at start
                # Keep prefix as empty string
                pass
            else:
                skipped_counts["empty_prefix"] += 1
                if verbose and len(debug_samples["empty_prefix"]) < 3:
                    debug_samples["empty_prefix"].append((idx, f"suffix={suffix[:50]}..." if suffix else "no suffix"))
                continue

        # Create messages using the FIM template
        messages = create_messages(prefix, suffix, sql_history, output_content)

        # Create the data item (no type field)
        data_item = {"messages": messages}

        hf_data.append(data_item)

    # Print sanitization info
    if sanitization_count > 0:
        print(f"\n  Sanitized {sanitization_count:,} rows (fixed unescaped quotes/invalid escapes in JSON)")

    # Print skip summary
    total_skipped = sum(skipped_counts.values())
    if total_skipped > 0:
        print(f"\n  Skipped {total_skipped:,} rows:")
        if skipped_counts["empty_input"] > 0:
            print(f"    - Empty/null INPUT: {skipped_counts['empty_input']:,}")
        if skipped_counts["empty_output"] > 0:
            print(f"    - Empty/null OUTPUT: {skipped_counts['empty_output']:,} (use --allow-empty-output to include)")
        if skipped_counts["json_parse_error"] > 0:
            print(f"    - JSON parse errors: {skipped_counts['json_parse_error']:,} (both JSON and regex extraction failed)")
        if skipped_counts["empty_prefix"] > 0:
            print(f"    - Empty prefix (<fillMe> at start): {skipped_counts['empty_prefix']:,}")

        # Show debug samples if verbose
        if verbose:
            for category, samples in debug_samples.items():
                if samples:
                    print(f"\n  Sample {category} rows:")
                    for row_idx, preview in samples:
                        print(f"    Row {row_idx}: {preview}...")

    print(f"\nConverted {len(hf_data):,} rows (from {len(df):,} total).              ")

    # Check if we have any valid data
    if len(hf_data) == 0:
        raise ValueError("No valid rows found after filtering. Check INPUT/OUTPUT columns for null/empty values.")

    # Create HuggingFace dataset
    print("\nCreating HuggingFace dataset...")
    hf_dataset = datasets.Dataset.from_list(hf_data)

    print(f"Dataset features: {hf_dataset.features}")
    print(f"Dataset size: {len(hf_dataset):,} examples")

    # Show sample
    print("\nSample entry:")
    sample = hf_dataset[0]
    print(f"  Messages: {len(sample['messages'])} messages")

    # System message (first message)
    system_content = sample["messages"][0]["content"]
    system_preview = system_content[:300] if len(system_content) > 300 else system_content
    print(f"  System content (first 300 chars): {system_preview}...")

    # User message (second message)
    user_content = sample["messages"][1]["content"]
    # Handle edge case where user_content might be shorter than expected
    user_preview_start = user_content[:500] if len(user_content) > 500 else user_content
    user_preview_end = user_content[-200:] if len(user_content) > 200 else user_content
    print(f"  User content (first 500 chars): {user_preview_start}...")
    print(f"  User content (last 200 chars): ...{user_preview_end}")

    # Assistant message (third message)
    print(f"  Assistant content: {sample['messages'][2]['content']}")

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
    parser = argparse.ArgumentParser(description="Convert SQL completion CSV/Parquet to HuggingFace dataset format")
    parser.add_argument("input_file", help="Path to the input CSV or Parquet file")
    parser.add_argument("-o", "--output", help="Output directory path (default: <input_basename>_hf_dataset)")
    parser.add_argument("--input-col", default="INPUT", help="Name of the input column (default: INPUT)")
    parser.add_argument("--output-col", default="OUTPUT", help="Name of the output column (default: OUTPUT)")
    parser.add_argument(
        "--format",
        choices=["disk", "parquet"],
        default="disk",
        help="Output format: 'disk' for HF save_to_disk, 'parquet' for parquet file (default: disk)",
    )
    parser.add_argument(
        "--allow-empty-output",
        action="store_true",
        help="Include rows with null/empty OUTPUT using '<no_suggestion>' token as assistant response",
    )
    parser.add_argument(
        "--skip-empty-prefix",
        action="store_true",
        help="Skip rows where <fillMe> is at the start (empty prefix). Default: include them.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed debug info for skipped rows",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CSV/Parquet to HuggingFace Dataset Converter")
    print("=" * 80)

    try:
        convert_to_hf_dataset(
            input_file_path=args.input_file,
            output_dir=args.output,
            input_col=args.input_col,
            output_col=args.output_col,
            save_format=args.format,
            allow_empty_output=args.allow_empty_output,
            allow_empty_prefix=not args.skip_empty_prefix,  # Default is to allow
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
