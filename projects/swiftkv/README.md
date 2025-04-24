# SwiftKV

The YAML files in this project provide example recipes for creating custom SwiftKV Llama models. Additionally, we share example resource requirements used in training these models. While the project includes a sample dataset, HuggingFaceH4/ultrachat_200k, we strongly recommend selecting datasets tailored to the specific tasks and use cases that matter most to you.

Please refer to our general [ArcticTraining quickstart](https://github.com/snowflakedb/ArcticTraining/tree/main?tab=readme-ov-file#quickstart) if you haven't already to understanding launching training jobs.

## meta-llama/Llama-3.1-8B-Instruct
* [swiftkv-llama-8b.yaml](configs/llama-3.1-swiftkv-8b-instruct.yaml)
* Training environment: 8 x H100 GPUs

## meta-llama/Llama-3.3-70B-Instruct
* [swiftkv-llama-70b.yaml](configs/llama-3.1-swiftkv-70b-instruct.yaml)
* Training environment: 64 X H100 GPUs

## meta-llama/Llama-3.1-405B-Instruct
* [swiftkv-llama-405b.yaml](configs/llama-3.1-swiftkv-405b-instruct.yaml)
* Training environment: 128 X H100 GPUs
