MODEL_PATH="/checkpoint/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693/"
OUTPUT_PATH="/checkpoint/speculator/llama-3.1-70b/5-heads/synth-data/Feb-5/weighted_sum"
DATASET_BASE="/checkpoint/users/samyam/datasets/synth/Feb-2/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8/"
DATASET1="\(PromptResponsePairs,${DATASET_BASE}/ultrachat/1K-gen/combined\)" 
DATASET2="\(PromptResponsePairs,${DATASET_BASE}/magicoder/1K-gen/combined\)"

CHECKPOINT_PATH=$OUTPUT_PATH/checkpoint
mkdir -p $OUTPUT_PATH
mkdir -p $CHECKPOINT_PATH

TRAIN_ITERS=3000

TRAIN_ARGS="--model_path $MODEL_PATH 
			--output_path $OUTPUT_PATH 
			--train_iters $TRAIN_ITERS 
			--datasets ${DATASET1} ${DATASET2}
			--zero_stage 3 
			--speculator_width 8192
			--checkpoint_interval 150
			--checkpoint_path $CHECKPOINT_PATH
			--micro_batch_size 1
			--global_batch_size 128
			--speculator_tie_weights
   			--speculator_scale_input
			--n_speculator_heads 5
			--mask_inputs
			--weighted_sum"

HOSTFILE="/data-fast/hostfile"

if [ -e ${HOSTFILE} ]; then
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/hf-hub'
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/temp'
	HF_HOME=/data-fast/hf-hub deepspeed -H ${HOSTFILE} train_speculator.py ${TRAIN_ARGS} &> ${OUTPUT_PATH}/train.log
else
	mkdir -p /data-fast/hf-hub
	mkdir -p /data-fast/temp
	HF_HOME=/data-fast/hf-hub deepspeed train_speculator.py ${TRAIN_ARGS} &> ${OUTPUT_PATH}/train.log
fi
