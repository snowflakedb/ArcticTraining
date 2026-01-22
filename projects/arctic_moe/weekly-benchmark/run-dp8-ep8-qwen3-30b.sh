#!/bin/bash
seq 20 | xargs -I -- echo

export TMPDIR=/data-fast/tmp
mkdir -p $TMPDIR

pkill -9 pt_main_thread
pkill -9 python
pkill -9 pt_data_worker

export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/checkpoint/huggingface
#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=0

/usr/bin/time -v arctic_training run-dp8-ep8-qwen3-30b.yml --num_gpus 8 --master_port 9828 -q |& tee run-dp8-ep8-qwen3-30b.log
#/usr/bin/time -v arctic_training --quiet run-dp8-ep8-qwen3-30b.yml |& tee run-dp8-ep8-qwen3-30b.log
