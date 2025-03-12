LOCAL_HOSTFILE='/data-fast/hostfile'

if [ -e ${HOSTFILE} ]; then
    HOSTFILE_TEMP='/code/users/samyam/hostfile_temp'
    cp ${LOCAL_HOSTFILE} ${HOSTFILE_TEMP}
    HOSTFILE=${HOSTFILE_TEMP}
fi

MODEL='neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8'
DATASET='magicoder-evol'
TENSOR_PARALLEL=1
OUTPUT_PATH="/checkpoint/users/samyam/datasets/synth/Feb-7/${MODEL}/${DATASET}/1K-gen"
BATCH_SIZE=5000

COMMAND="python /code/users/samyam/ArcticTraining/projects/mlp_speculator/mlp_speculator/data_generation/data_generation.py
        --hf_dataset ${DATASET}
        --model ${MODEL}
        --tensor_parallel ${TENSOR_PARALLEL}
        --batch_size ${BATCH_SIZE}
        --hf_dataset ${DATASET}
        --output_dataset_path ${OUTPUT_PATH}
        --max_tokens 1024
        --skip_launch
        "
SET_ENV="source /code/users/samyam/snowflakedb-vllm/vllm/venv/bin/activate"

COMMAND_DIST="${COMMAND} --hostfile ${HOSTFILE}"

if [ -e ${HOSTFILE} ]; then
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/hf-hub'
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/temp'
    echo "Running: ${COMMAND_DIST}"
	ds_ssh -f ${HOSTFILE} "${SET_ENV}; HF_HOME=/data-fast/hf-hub ${COMMAND_DIST}"
else
	mkdir -p /data-fast/hf-hub
	mkdir -p /data-fast/temp
    HF_HOME=/data-fast/hf-hub ${COMMAND}
fi
