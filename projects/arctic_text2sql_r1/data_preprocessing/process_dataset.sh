#!/usr/bin/env bash
set -e

usage() {
  cat <<EOF
Usage: $0 \
  -i INPUT_JSON       path to input_data_file \
  -o OUTPUT_JSON      path to output_data_file \
  -d DB_PATH          path to the directory of databases \
  -t TABLES_JSON      path to tables definition JSON \
  -s SOURCE           data source name (e.g. "bird") \
  -m MODE             dataset mode (e.g. "dev", "train") \
  -v VALUE_LIMIT_NUM  integer limit for values \
  -c DB_INDEX_PATH    path to db_content_index

Example:
  $0 \
    -i /data-fast/bird/dev_20240627/dev.json \
    -o /data-fast/dev_bird.json \
    -d /data-fast/bird/dev_20240627/dev_databases/ \
    -t ./data/bird/dev_20240627/dev_tables.json \
    -s bird \
    -m dev \
    -v 2 \
    -c ./data/bird/dev_20240627/db_contents_index
EOF
  exit 1
}

while getopts "i:o:d:t:s:m:v:c:" opt; do
  case $opt in
    i) INPUT_JSON=$OPTARG ;;
    o) OUTPUT_JSON=$OPTARG ;;
    d) DB_PATH=$OPTARG ;;
    t) TABLES_JSON=$OPTARG ;;
    s) SOURCE=$OPTARG ;;
    m) MODE=$OPTARG ;;
    v) VALUE_LIMIT_NUM=$OPTARG ;;
    c) DB_INDEX=$OPTARG ;;
    *) usage ;;
  esac
done

# make sure all required args are provided
: "${INPUT_JSON:?Missing -i INPUT_JSON}"
: "${OUTPUT_JSON:?Missing -o OUTPUT_JSON}"
: "${DB_PATH:?Missing -d DB_PATH}"
: "${TABLES_JSON:?Missing -t TABLES_JSON}"
: "${SOURCE:?Missing -s SOURCE}"
: "${MODE:?Missing -m MODE}"
: "${VALUE_LIMIT_NUM:?Missing -v VALUE_LIMIT_NUM}"
: "${DB_INDEX:?Missing -c DB_INDEX}"

python data_preprocessing/process_dataset.py \
  --input_data_file "$INPUT_JSON" \
  --output_data_file "$OUTPUT_JSON" \
  --db_path "$DB_PATH" \
  --tables "$TABLES_JSON" \
  --source "$SOURCE" \
  --mode "$MODE" \
  --value_limit_num "$VALUE_LIMIT_NUM" \
  --db_content_index_path "$DB_INDEX"
