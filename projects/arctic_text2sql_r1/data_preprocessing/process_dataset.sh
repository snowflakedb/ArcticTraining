set -e

# BIRD (dev)
python process_dataset.py --input_data_file ./data/bird/dev_20240627/dev.json --output_data_file ./data/dev_bird.json --db_path ./data/bird/dev_20240627/dev_databases/ --tables ./data/bird/dev_20240627/dev_tables.json --source bird --mode dev --value_limit_num 2 --db_content_index_path ./data/bird/dev_20240627/db_contents_index
