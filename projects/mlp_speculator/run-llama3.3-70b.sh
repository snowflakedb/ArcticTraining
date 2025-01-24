MODEL_PATH="/checkpoint/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/38ff4e01a70559264c95945aa04b900a11e68422/"
OUTPUT_PATH="/checkpoint/speculator/llama-3.3-70b/5-heads/Jan-14"
CHECKPOINT_PATH=$OUTPUT_PATH/checkpoint
mkdir -p $OUTPUT_PATH
mkdir -p $CHECKPOINT_PATH

TRAIN_ITERS=3000

TRAIN_ARGS="--model_path $MODEL_PATH 
			--output_path $OUTPUT_PATH 
			--train_iters $TRAIN_ITERS 
			--zero_stage 3 
			--gen_train 
			--speculator_width 8192
			--checkpoint_interval 300
			--checkpoint_path $CHECKPOINT_PATH
			--gen_micro_batch 320
			--micro_batch_size 5
			--speculator_tie_weights
   			--speculator_scale_input
			--n_speculator_heads 5"

HOSTFILE="/data-fast/hostfile"

if [ -e ${HOSTFILE} ]; then
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/hf-hub'
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/temp'
	ds_ssh -f ${HOSTFILE} 'HF_HOME=/data-fast/hf-hub python /code/users/samyam/ArcticTraining/projects/mlp_speculator/create_datasets.py'
	HF_HOME=/data-fast/hf-hub deepspeed -H ${HOSTFILE} train_speculator.py ${TRAIN_ARGS} &> ${OUTPUT_PATH}/train.log
else
	mkdir -p /data-fast/hf-hub
	mkdir -p /data-fast/temp
	HF_HOME=/data-fast/hf-hub python create_datasets.py
    HF_HOME=/data-fast/hf-hub deepspeed train_speculator.py ${TRAIN_ARGS} &> ${OUTPUT_PATH}/train.log
fi
