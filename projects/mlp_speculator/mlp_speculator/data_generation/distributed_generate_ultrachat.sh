HOSTFILE='/data-fast/hostfile'
MODEL='neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8'
DATASET='ultrachat'
OUTPUT_PATH="/checkpoint/users/samyam/datasets/synth/${MODEL}/${DATASET}"
TENSOR_PARALLEL=2
BATCH_SIZE=2000


COMMAND="python /code/users/samyam/ArcticTraining/projects/mlp_speculator/mlp_speculator/data_generation/data_generation.py
        --hf_dataset ${DATASET}
        --model ${MODEL}
        --tensor_parallel ${TENSOR_PARALLEL}
        --batch_size ${BATCH_SIZE}
        --hf_dataset ${DATASET}
        --output_dataset_path ${OUTPUT_PATH}
        "

if [ -e ${HOSTFILE} ]; then
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/hf-hub'
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/temp'
	ds_ssh -f ${HOSTFILE} 'HF_HOME=/data-fast/hf-hub ${COMMAND} --hostfile ${HOSTFILE}'
else
	mkdir -p /data-fast/hf-hub
	mkdir -p /data-fast/temp
    HF_HOME=/data-fast/hf-hub ${COMMAND}
fi
