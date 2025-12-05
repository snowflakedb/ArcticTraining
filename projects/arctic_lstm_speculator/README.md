# ArcticLSTMSpeculator Training

This directory contains the necessary code and scripts to train an LSTM-based speculator for Arctic inference. The training process involves generating a synthetic dataset using a base model and then training the speculator on this data.

## Quick Start

Before starting, ensure you have set up the general [ArcticTraining environment](https://github.com/snowflakedb/ArcticTraining/tree/main?tab=readme-ov-file#quickstart).

To launch an end-to-end workflow (Data Generation $\rightarrow$ Training $\rightarrow$ Inference Serving), use the provided bash scripts located in the `scripts/` directory.

```bash
# Example: Training a speculator for Llama-3.1-8B-Instruct
bash scripts/llama3.1-8b.sh
```

## 1\. End-to-End Training Scripts

The scripts in `scripts/*.sh` control the entire pipeline. You can modify the control variables at the top of the script to skip stages or change parallelism:

  * **`DATA_GEN=1`**: Set to `1` to generate training data using vLLM. Set to `0` if you already have data.
  * **`TRAIN=1`**: Set to `1` to run the `arctic_training` job.
  * **`vllm_tensor_parallel`**: Number of GPUs to use for the teacher model during data generation.
  * **`total_num_of_scripts`**: Number of parallel data generation jobs to spawn (usually matches your GPU count).

The script automatically pulls configuration details (like model name and output directories) from the associated YAML config file.

## 2\. Customizing Data Generation

The data generation process uses a teacher model to generate responses to prompts, which serves as the training data for the speculator. The core logic is in `speculator/data_generation/vllm_data_generation.py`.

### modifying Generation Arguments

You can customize generation parameters by modifying the `data_gen_script_maker.py` call in your bash script or directly changing defaults in `vllm_data_generation.py`:

  * `--hf_dataset`: The source dataset for prompts (default: `ultrachat`).
  * `--max_tokens`: Maximum tokens to generate per response (default: `256`).
  * `--gen_temp`: Sampling temperature (default: `1.0`).
  * `--tensor_parallel`: Tensor parallelism for the vLLM engine.

### Adding New Datasets

To use a custom dataset, you must modify the `load_hf_dataset` function in `speculator/data_generation/vllm_data_generation.py`. Add a new condition for your dataset name:


```python
def load_hf_dataset(dataset):
    if dataset == "my_custom_dataset":
        # Load your dataset
        result = load_dataset("my/dataset/path", split="train")

        def format_fn(example):
             return {
                "messages": [
                    {"role": "user", "content": example["my_prompt_col"]},
                    {"role": "assistant", "content": example["my_response_col"]}
                ]
            }
        return result.map(format_fn)
    # ... existing datasets ...
```

After modifying the code, update the `--hf_dataset` argument in your script (e.g., `data_gen_script_maker.py`) to matches your new condition (e.g., `"my_custom_dataset"`).

## 3\. Customizing Training Configuration

Training is configured via YAML files (e.g., `llama3.1-8b.yaml`). You can create a new YAML file or modify an existing one to change model architecture or training hyperparameters.

### Key Configuration Fields

#### Model Section (`model`)

This section defines the speculator architecture:

  * **`name_or_path`**: The base Hugging Face model (e.g., `meta-llama/Llama-3.1-8B-Instruct`).
  * **`n_speculator_heads`**: Number of future tokens the speculator predicts (lookahead depth).
  * **`speculator_width`**&**`proj_dim`** & **`emb_dim`**: Hidden dimension size of the LSTM speculator.
  * **`method`**: Architecture type (default: `sum_lstm`).

#### Data Section (`data`)

  * **`sources`**: List of data sources. Set `name_or_path` to the folder containing your generated data (e.g., `llama31_8b_data`).

#### Checkpoint (`checkpoint`)

Defines where to save the model:

  * **`output_dir`**: Path to save the final speculator weights (e.g., `spec-decode-llama31-8b`).

## 4\. Serving / Inference

The bash script includes a command to serve the model using `vllm` immediately after training. It enables speculative decoding with the trained draft model:

```bash
export VLLM_USE_V1=1
vllm serve $model_name \
    --speculative-config "{\"model\": \"$spec_drafter_name\", ... \"method\": \"arctic\"}" \
    ...
```

Ensure `ArcticInference` is installed and set up correctly for this step.