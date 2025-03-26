# ExCoT-DPO Project: Training and Evaluation

This repository provides both demo training and evaluation setups for our ExCoT-DPO project. The first section covers launching a DPO demo training job, and the second section explains how to evaluate ExCoT-DPO on the BIRD and SPIDER datasets.


## Training Setup: DPO Demo Training Job

To launch a DPO demo training job, follow these steps:

1. **Install the Arctic Training Library:**

   Please refer to the root directory README for detailed instructions on installing the Arctic Training Library.

2. **Launch the Demo Training Job:**

   Execute the following command to start a demo training job using the provided configuration file:

   ```bash
   arctic_training projects/dpo/dpo-llama-8b.yaml
   ```
   This command will initiate the training process as defined in the dpo-llama-8b.yaml configuration file.

## Evaluation Setup: ExCoT-DPO on BIRD and SPIDER

This section provides instructions for setting up your environment to evaluate ExCoT-DPO models on the BIRD and SPIDER datasets.

### Requirements

- Python 3.8+
- pip (comes with Python)
- [Optional] Conda (if you choose to use a Conda environment)

### Environment Setup

#### Using Python Virtual Environment

1. **Create a Virtual Environment:**

    ```bash
    python -m venv /data-fast/vllm-venv
    ```
2. **Activate the Environement:**
    ```bash
    source /data-fast/vllm-venv/bin/activate
    ```
3. **Install the Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
#### Using Conda Environment
```bash
conda env create -f environment.yml
conda activate dpo
pip install -r requirements.txt
```

### Evaluation Instructions
To evaluate the ExCoT-DPO models on BIRD and SPIDER, follow these steps:

#### Dataset Download

1. **Download:**
    - Download the BIRD dataset from [BIRD Dataset](https://bird-bench.github.io/)
2. **Configuration:**
    - Update the [bird_config.yaml](dpo/evaluation/configs/bird_config.yaml) file with your desired evaluation settings, ensuring that it points to the correct datasets.
3. **Run the Evaluation Script:**
    Execute the evaluation script with the configuration file:
    ```bash
    python eval_w_arctic_syth.py --model-name MODEL_NAME --prompt-version divide-and-conquer --mode dev --task-name EVAL_TASK_NAME --data-config ./configs/bird_config.yaml
    ```
    ```bash
    python eval_w_arctic_syth.py --model-name MODEL_NAME --prompt-version divide-and-conquer --mode test --task-name EVAL_TASK_NAME --data-config ./configs/spider_config.yaml
    ```
