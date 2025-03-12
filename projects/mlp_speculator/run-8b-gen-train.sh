MODEL_PATH="/checkpoint/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
OUTPUT_PATH="/checkpoint/speculator/llama-3.1-8b/Feb-7-gen-train/"
CHECKPOINT_PATH=$OUTPUT_PATH/checkpoint
mkdir -p $OUTPUT_PATH
mkdir -p $CHECKPOINT_PATH


TRAIN_ITERS=3000

TRAIN_ARGS="--model_path $MODEL_PATH 
			--output_path $OUTPUT_PATH 
			--train_iters $TRAIN_ITERS  
			--checkpoint_path $CHECKPOINT_PATH
			--zero_stage 3 
			--gen_train  
			--micro_batch_size 12
			--checkpoint_interval 300
			--gen_micro_batch 768
			--gen_train_global_batch 2048
    		--gen_train_micro_batch 32"

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
