# ExCoT-DPO Project: Training and Evaluation
[[🤗Llama-3.1-Arctic-ExCoT-70B](https://huggingface.co/Snowflake/Llama-3.1-Arctic-ExCoT-70B)] [[🤗Qwen-2.5-coder-Arctic-ExCoT-32B](https://huggingface.co/Snowflake/Qwen-2.5-coder-Arctic-ExCoT-32B)]

This repository includes demo setups for both training and evaluation of our ExCoT-DPO project.
We begin by covering SFT and DPO data generation.
Next, we provide demo scripts for running one SFT and one DPO training example.
Finally, we demonstrate how to evaluate ExCoT-DPO on the BIRD and SPIDER datasets.


## Data Generation: SFT & DPO data generation

To launch a data generation job, follow these steps:

1. **Install the Arctic Training Library:**

   Please refer to the root directory README for detailed instructions on installing the Arctic Training Library.

2. **Install BIRD and Spider datasets:**

    BIRD benchmark installation links: [train](https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip), [dev](https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip)

    Spider benchmark installation links: [dataset](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view)


3. **Data Generation:**

    Update configs with your data locations
    ```ArcticTraining/projects/excot_dpo/data_generation/configs/bird_config.yaml```
    ```ArcticTraining/projects/excot_dpo/data_generation/configs/spider_config.yaml```

    Set up API and Token for GPT based data generation
    ```bash
    export AZURE_OPENAI_API_KEY=
    export AZURE_OPENAI_ENDPOINT=
    ```

    Launch GPT/vLLM based data generation, in ExCoT we use GPT based generation for SFT and Off-Policy DPO
    ```
    python data_generation/data_generation.py \
        --config-path data_generation/configs/bird_config.yaml \
        --type gpt
    ```
    or
    ```
    python data_generation/data_generation.py \
        --config-path data_generation/configs/bird_config.yaml \
        --type vllm \
        --model-name MODEL_NAME \
        --tp-size 8
    ```
    After data generation we can launch data_verification
    ```
    python data_generation/local_verificaiton.py \
        --config-path data_generation/configs/bird_config.yaml \
        --gpt-cot-path YOUR_GPT_GEN_PATH/results.jsonl \
        --output-path OUTPUT_PATH
    ```
    After data generation, we can sample the correct data for SFT or DPO

    ``` sft_sample.py ``` and ``` dpo_sample.py ``` are two useful scripts to call.

4. Training
