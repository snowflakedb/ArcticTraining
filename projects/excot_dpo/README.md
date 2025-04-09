# ExCoT-DPO Project: Training and Evaluation

This repository provides a complete demo setup for training and evaluating **ExCoT-DPO**, our framework for improved instruction tuning using explicit chain-of-thought (ExCoT) and direct preference optimization (DPO). For more details see our [arxiv paper](https://arxiv.org/pdf/2503.19988) and blog.

🚀 Try our released models on Hugging Face:
- [🤗 Llama-3.1-Arctic-ExCoT-70B](https://huggingface.co/Snowflake/Llama-3.1-Arctic-ExCoT-70B)
- [🤗 Qwen-2.5-coder-Arctic-ExCoT-32B](https://huggingface.co/Snowflake/Qwen-2.5-coder-Arctic-ExCoT-32B)

### What's Inside

- **Data Preparation**: Scripts for generating data used in supervised fine-tuning (SFT) and DPO.
- **Training Examples**: Simple, runnable scripts for:
  - One SFT training run
  - One DPO training run
- **Evaluation**: Instructions for evaluating ExCoT-DPO on two benchmarks:
  - **BIRD**: Benchmark for Instruction-following with Reasoning and Dialogue
  - **SPIDER**: Complex text-to-SQL generation benchmark


## Data Generation

This section covers how to generate training data for both **Supervised Fine-Tuning (SFT)** and **Direct Preference Optimization (DPO)**.

### 1. Install the Arctic Training Library

Please refer to the root-level `README.md` for detailed setup instructions.

Some extra packages need to install
```bash
pip install sqlglot
pip install editdistance
```

### 2. Download Datasets

Download the following datasets and extract them into a single directory. For the purposes of this tutorial we will assume they are all extracted into `/data/`. If your path differs please adjust the config files under [data_generation/configs/bird_config.yaml](data_generation/configs/bird_config.yaml) and [data_generation/configs/spider_config.yaml](data_generation/configs/spider_config.yaml).

- **BIRD Benchmark**
  - [Train Set](https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip)
  - [Dev Set](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip)

- **SPIDER Benchmark**
  - [Dataset (Google Drive)](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view)

### 3. Launch Data Generation with Azure OpenAI endpoints

In our ExCoT paper we used Azure OpenAI for SFT and off-policy DPO data generation. In order to replicate this work you will need to use this endpoint and set your GPT API credentials accordingly:

```bash
export AZURE_OPENAI_API_KEY=<your-key>
export AZURE_OPENAI_ENDPOINT=<your-endpoint>
```

**Azure OpenAI GPT-based (default for ExCoT):**
```bash
python data_generation/data_generation.py \
    --config-path data_generation/configs/bird_config.yaml \
    --gpt-output-path YOUR_GPT_GEN_PATH \
    --data-task-name demo \
    --type gpt
```

For on-policy DPO generated data we must do this from a local model hosted by vLLM which can be executed accordingly:

**vLLM-based**
```bash
python data_generation/data_generation.py \
    --config-path data_generation/configs/bird_config.yaml \
    --type vllm \
    --model-name MODEL_NAME \
    --vllm-output-path VLLM_VERIFIED_DATASET_OUTPUT_PATH \
    --tp-size 8
```

### 4. Verify Generated Data
After generation, run verification.

** Verify SFT and off-policy DPO data **

```bash
python data_generation/local_verification.py \
    --config-path data_generation/configs/bird_config.yaml \
    --gpt-cot-path YOUR_GPT_GEN_PATH/demo/results.jsonl \
    --output-path VERIFIED_SFT_PATH
```
** Verify on-policy DPO data **

```bash
python data_generation/local_verification.py \
    --config-path data_generation/configs/bird_config.yaml \
    --vllm-cot-path VLLM_VERIFIED_DATASET_OUTPUT_PATH \
    --output-path VERIFIED_DPO_PATH
```

### 5. Sample Data for SFT and DPO
Use the following scripts to prepare your datasets:
* For SFT and off-policy DPO:
```bash
python data_generation/sft_sample.py --verify-path VERIFIED_SFT_PATH --output-path OUTPUT_PATH
```
* For on-policy DPO:
```bash
python data_generation/dpo_sample.py --dataset-path VERIFIED_DPO_PATH --version VERSION --output-path OUTPUT_PATH
```
✅ Once your data is ready, you can move on to model training!

## Training and Evaluating ExCoT-DPO Models

### 1. SFT & DPO Training

We provide example YAML configuration files for launching both SFT and DPO training runs:

- `sft-llama-8b.yaml`
- `dpo-llama-8b.yaml`

You can customize these YAML files to adjust model paths, batch sizes, learning rates, and other parameters to suit your setup.

To launch training jobs, use:

```bash
arctic_training sft-llama-8b.yaml
```
or
```bash
arctic_training dpo-llama-8b.yaml
```
Make sure you’ve installed the Arctic training CLI correctly and that your environment is set up.


### 2. Evaluate Model Checkpoints

You can evaluate either:

- A locally trained checkpoint
- Or a released model from the HuggingFace model cards

Please also update ```configs/bird_config.yaml``` and ```configs/spider_config.yaml```


Run the following script:

```bash
python eval_w_arctic_syth.py \
    --model-name {YOUR_MODEL} \
    --data-config configs/bird_config.yaml \
    --prompt-version "divide-and-conquer" \
    --mode dev \
    --task-name bird_eval
```

For Spider evaluation, simply change the ```--mode``` flag and point to the corresponding config file.

✅ Evaluation results will be saved under the task name directory.
