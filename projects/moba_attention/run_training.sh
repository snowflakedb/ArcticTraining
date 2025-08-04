#!/bin/bash
mkdir /data-fast/tmp
HF_HOME=/checkpoint/huggingface/ HF_DATASETS_CACHE=/data-fast/tmp TMPDIR=/data-fast/tmp/ arctic_training moba-swiftkv-llama-8b-long-sequence.yaml
