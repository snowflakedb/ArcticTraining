MODEL='neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8'
DATASET=$1
OUTPUT_PATH="/checkpoint/users/samyam/datasets/synth/Feb-7/${MODEL}/${DATASET}/1K-gen"

python -c "from data_generation import combine_datasets; combine_datasets('${OUTPUT_PATH}')"
