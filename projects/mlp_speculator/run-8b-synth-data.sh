MODEL_PATH="/checkpoint/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"
OUTPUT_PATH="/checkpoint/speculator/llama-3.1-8b/Feb-7-synth-data-extended-no-mask-no-evol"
DATASET_BASE="/checkpoint/users/samyam/datasets/synth/Feb-7/neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8/"
DATASET1="\(PromptResponsePairs,${DATASET_BASE}/ultrachat/1K-gen-with-extended-prompts/combined\)" 
DATASET2="\(PromptResponsePairs,${DATASET_BASE}/magicoder/1K-gen/combined\)"
DATASET3="\(PromptResponsepairs,${DATASET_BASE}/magicoder-evol/1K-gen/combined\)"

CHECKPOINT_PATH=$OUTPUT_PATH/checkpoint
mkdir -p $OUTPUT_PATH
mkdir -p $CHECKPOINT_PATH

TRAIN_ITERS=3000

TRAIN_ARGS="--model_path $MODEL_PATH 
			--output_path $OUTPUT_PATH 
			--train_iters $TRAIN_ITERS
			--datasets ${DATASET1} ${DATASET2}
			--checkpoint_path $CHECKPOINT_PATH
			--zero_stage 3 
			--checkpoint_interval 300
			--micro_batch_size 2
			--global_batch_size 128
			"
#--mask_inputs
			
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
