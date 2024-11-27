MODEL_PATH="/checkpoint/swiftkv/llama-swiftkv-70b-20240914/swiftkv_renamed"
OUTPUT_PATH="/checkpoint/speculator/llama-3.1-swiftkv-70b/Nov-27-6-43"
mkdir -p $OUTPUT_PATH
TRAIN_ITERS=3000

TRAIN_ARGS="--model_path $MODEL_PATH 
			--output_path $OUTPUT_PATH 
			--train_iters $TRAIN_ITERS 
			--zero_stage 3 
			--gen_train 
			--checkpoint_interval 300
			--speculator_tie_weights
   			--speculator_scale_input"

HOSTFILE="/data-fast/hostfile"

if [ -e ${HOSTFILE} ]; then
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/hf-hub'
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/temp'
	ds_ssh -f ${HOSTFILE} 'HF_HOME=/data-fast/hf-hub python /code/users/samyam/ArcticTraining/projects/mlp_speculator/create_datasets.py'
	HF_HOME=/data-fast/hf-hub deepspeed -H ${HOSTFILE} train_speculator.py ${TRAIN_ARGS} &> ${OUTPUT_PATH}/train.log&
else
	mkdir -p /data-fast/hf-hub
	mkdir -p /data-fast/temp
	HF_HOME=/data-fast/hf-hub python create_datasets.py
    HF_HOME=/data-fast/hf-hub deepspeed train_speculator.py ${TRAIN_ARGS} &> ${OUTPUT_PATH}/train.log&
fi
