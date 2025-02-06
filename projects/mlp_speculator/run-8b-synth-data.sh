MODEL_PATH="/checkpoint/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"
OUTPUT_PATH="/checkpoint/speculator/llama-3.1-8b/5-heads/synth-data/Jan-25"
DATASET_BASE="/checkpoint/users/samyam/datasets/synth/neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8/"
DATASET1="(PromptResponsePairs,${DATASET_BASE}/ultrachat/combined)" 
DATASET2="(PromptResponsePairs,${DATASET_BASE}/magicoder/combined)"

CHECKPOINT_PATH=$OUTPUT_PATH/checkpoint
mkdir -p $OUTPUT_PATH
mkdir -p $CHECKPOINT_PATH

TRAIN_ITERS=3000

TRAIN_ARGS="--model_path $MODEL_PATH 
			--output_path $OUTPUT_PATH 
			--train_iters $TRAIN_ITERS 
			--datasets ${DATASET1} ${DATASET2}
			--zero_stage 3 
			--speculator_width 4096
			--checkpoint_interval 300
			--checkpoint_path $CHECKPOINT_PATH
			--micro_batch_size 1
			--global_batch_size 128
			--speculator_tie_weights
   			--speculator_scale_input
			--n_speculator_heads 3
			--mask_inputs"

HOSTFILE="/data-fast/hostfile"

if [ -e ${HOSTFILE} ]; then
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/hf-hub'
	ds_ssh	-f ${HOSTFILE} 'mkdir -p /data-fast/temp'
	HF_HOME=/data-fast/hf-hub deepspeed -H ${HOSTFILE} train_speculator.py ${TRAIN_ARGS} #&> ${OUTPUT_PATH}/train.log
else
	mkdir -p /data-fast/hf-hub
	mkdir -p /data-fast/temp
	HF_HOME=/data-fast/hf-hub deepspeed train_speculator.py ${TRAIN_ARGS} #&> ${OUTPUT_PATH}/train.log
fi
