#!/bin/bash

export PYTORCH_ALLOC_CONF=expandable_segments:True
#export HF_HOME=/data/huggingface

/usr/bin/time -v arctic_training run-qwen3-32b.yml --num_gpus 1 --master_port 9828 -q |& tee run-qwen3-32b.log
