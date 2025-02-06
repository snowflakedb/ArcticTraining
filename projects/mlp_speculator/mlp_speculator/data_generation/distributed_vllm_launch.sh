LOCAL_HOSTFILE='/data-fast/hostfile'

if [ -e ${LOCAL_HOSTFILE} ]; then
    HOSTFILE_TEMP='/code/users/samyam/hostfile_temp'
    cp ${LOCAL_HOSTFILE} ${HOSTFILE_TEMP}
    HOSTFILE=${HOSTFILE_TEMP}
fi

MODEL='neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8'
TENSOR_PARALLEL=2

COMMAND="python /code/users/samyam/ArcticTraining/projects/mlp_speculator/mlp_speculator/data_generation/data_generation.py
        --model ${MODEL}
        --tensor_parallel ${TENSOR_PARALLEL}
        --skip_generation
        "
SET_ENV="source /code/users/samyam/snowflakedb-vllm/vllm/venv/bin/activate"

COMMAND_DIST="${COMMAND} --hostfile ${HOSTFILE}"

if [ -e ${HOSTFILE} ]; then
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/hf-hub'
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/temp'
    echo "Testing: ${COMMAND_DIST}"
	ds_ssh -f ${HOSTFILE} "${SET_ENV}; HF_HOME=/data-fast/hf-hub ${COMMAND_DIST}"
else
	mkdir -p /data-fast/hf-hub
	mkdir -p /data-fast/temp
    HF_HOME=/data-fast/hf-hub ${COMMAND}
fi
