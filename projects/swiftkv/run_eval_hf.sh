set -ex
FINAL=${1}

OUTPUT=`basename ${FINAL}`
mkdir $OUTPUT

export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export HF_HOME=/data-fast/hf-home
export HF_HOME=/checkpoint/huggingface

# EVAL_CMD=$(cat <<EOF
# python eval.py \
#   --model hf \
#   --model_args pretrained=${MODEL},dtype=auto,max_model_len=4096,tensor_parallel_size=4 \
#   --gen_kwargs max_gen_toks=1024 \
#   --batch_size auto \
#   --log_samples \
#   --output_path ${MODEL}
# EOF
# )

EVAL_CMD=$(cat <<EOF
accelerate launch eval.py \
  --model hf \
  --model_args pretrained=${FINAL},add_bos_token=True \
  --batch_size 8 \
  --log_samples \
  --output_path ${FINAL}
EOF
)

# arc challenge
${EVAL_CMD} \
  --tasks arc_challenge_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 2>&1 | tee ${OUTPUT}/arc.log

# winogrande

# helloswag

# truthfulqa

# mmlu
${EVAL_CMD} \
  --tasks mmlu_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5 2>&1 | tee ${OUTPUT}/mmlu.log

# mmlu cot
${EVAL_CMD} \
  --tasks mmlu_cot_0shot_llama_3.1_instruct \
  --gen_kwargs max_gen_toks=512 \
  --apply_chat_template \
  --num_fewshot 0 2>&1 | tee ${OUTPUT}/mmlu_cot.log

# gsm8k
${EVAL_CMD} \
  --tasks gsm8k_cot_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 8 2>&1 | tee ${OUTPUT}/gsm8k.log
