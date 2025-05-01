# Tested with 8xH100 node

DATA_GEN=1
TRAIN=1
data_save_folder_name="llama31_70b_tmp"
vllm_tensor_parallel=4
script_save_path="llama31_70b_data_gen_scripts"
total_num_of_scripts=8

train_config_file="llama3.1-70b.yaml"
# must be aligned with the config file
model_name=$(yq .model.name_or_path $train_config_file)
data_concat_folder_name=$(yq '.data.sources[0].name_or_path' $train_config_file)
spec_drafter_name=$(yq '.checkpoint[1].output_dir' $train_config_file)
num_speculative_tokens=$(yq .model.n_speculator_heads $train_config_file)


# data generation
if [ "$DATA_GEN" -eq "1" ]; then
  pip install vllm
  rm -r $script_save_path ${script_save_path}_tmp
  python speculator/data_generation/data_gen_script_maker.py --model_name=$model_name \
    --data_save_folder_name=$data_save_folder_name \
    --vllm_tensor_parallel=$vllm_tensor_parallel \
    --script_save_path=$script_save_path \
    --total_num_of_scripts=$total_num_of_scripts
  python multigpu_runner.py $script_save_path --max_gpus=8 -n $vllm_tensor_parallel
  python speculator/data_generation/concat_generated_datasets.py --data_save_folder_name=$data_save_folder_name --data_concat_folder_name=$data_concat_folder_name
fi

# run ArcticTraining
if [ "$TRAIN" -eq "1" ]; then
  arctic_training $train_config_file
fi


export VLLM_USE_V1=1
vllm serve $model_name \
    --disable-log-requests \
    --tensor-parallel-size 8 \
    --enable-chunked-prefill \
    --speculative-config "{\"model\": \"$spec_drafter_name\", \"num_speculative_tokens\": $num_speculative_tokens, \"draft_tensor_parallel_size\": 8, \"method\": \"arctic\"}" \
    --gpu_memory_utilization 0.9
