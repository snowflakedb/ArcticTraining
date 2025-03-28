FINAL=${1}

export HF_HOME=/data-fast/hf-home

EVAL_CMD=$(cat <<EOF
accelerate launch eval.py \
  --model hf \
  --model_args pretrained=${FINAL},add_bos_token=True,attn_implementation=flash_attention_2 \
  --gen_kwargs max_gen_toks=1024 \
  --log_samples \
  --output_path ${FINAL}
EOF
)

${EVAL_CMD} \
  --tasks truthfulqa_mc2 \
  --batch_size 16 \
  --num_fewshot 0

${EVAL_CMD} \
  --tasks winogrande \
  --batch_size 16 \
  --num_fewshot 5

${EVAL_CMD} \
  --tasks hellaswag \
  --batch_size 4 \
  --num_fewshot 10

${EVAL_CMD} \
  --tasks arc_challenge_llama_3.1_instruct \
  --batch_size 16 \
  --apply_chat_template \
  --num_fewshot 0

${EVAL_CMD} \
  --tasks gsm8k_cot_llama_3.1_instruct \
  --batch_size 4 \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 8

${EVAL_CMD} \
  --tasks mmlu_llama_3.1_instruct \
  --batch_size 1 \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5

${EVAL_CMD} \
  --tasks mmlu_cot_0shot_llama_3.1_instruct \
  --batch_size 8 \
  --apply_chat_template \
  --num_fewshot 0
