[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/snowflakedb/ArcticTraining/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/arctic-training.svg)](https://pypi.org/project/arctic-training/)

<h3 align="center">
  <img src="docs/images/arctic_training_logo.svg" width=500px><br>
  | <a href="https://arctictraining.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://www.snowflake.com/en/engineering-blog/arctictraining-llm-post-training-framework/"><b>Blog</b></a> |
</h3>

<!--| <a href="#"><b>Discourse</b></a> | -->

## Latest News

* [2025/05] [Fastest Speculative Decoding in vLLM with Arctic Inference and Arctic Training](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)
* [2025/03] [Snowflake Arctic Embed Joins ArcticTraining: Simple And Scalable Embedding Model Training](https://www.snowflake.com/en/engineering-blog/arctic-embed-joins-arctictraining/)
* [2025/01] [ArcticTraining: Simplifying and Accelerating Post-Training for LLMs](https://www.snowflake.com/en/engineering-blog/arctictraining-llm-post-training-framework/)
* [2024/12] [SwiftKV: Accelerating Enterprise LLM Workloads with Knowledge Preserving Compute Reduction](https://www.snowflake.com/en/engineering-blog/swiftkv-llm-compute-reduction/)

# ArcticTraining: Simplifying and Accelerating Post-Training for LLMs

ArcticTraining is a framework designed to simplify and accelerate the post-training process for large language models (LLMs). It addresses challenges in current frameworks, such as limited support for rapid prototyping and the lack of native data generation tools, by offering modular trainer designs, simplified code structures, and integrated pipelines for creating and cleaning synthetic data. These features enable users to enhance LLM capabilities, like code generation and complex reasoning, with greater efficiency and flexibility. Read more about ArcticTraining [in our blog](https://www.snowflake.com/en/engineering-blog/arctictraining-llm-post-training-framework/).

# Projects

The projects folder contains various special projects we have released that build on-top of ArcticTraining. Each project includes it's own README and associated assets to get started:

* [SwiftKV](projects/swiftkv)
* [Speculative Decoding](projects/mlp_speculator)
* [Arctic-Embed](projects/arctic_embed)

# Quickstart

To get started training a model with ArcticTraining, follow the steps below:

1. Install the ArcticTraining package and its dependencies:

```bash
pip install arctic-training
```

2. Create a training recipe YAML that uses the built-in Supervised Fine-Tuning (SFT) trainer:

```yaml
type: sft
micro_batch_size: 2
model:
  name_or_path: meta-llama/Llama-3.1-8B-Instruct
data:
  sources:
    - HuggingFaceH4/ultrachat_200k
checkpoint:
  - type: huggingface
    save_end_of_training: true
    output_dir: ./fine-tuned-model
```

3. Run the training recipe with the ArcticTraining CLI (see below). This will use the `DeepSpeed` launcher behind the scenes, you can pass any compatible DeepSpeed launcher arguments to the ArcticTraining CLI (e.g., --num_nodes, --num_gpus).

```bash
arctic_training path/to/sft-recipe.yaml
```

## Customize Training

To customize the training workflow, you can modify the training recipe YAML we
created in step 3 above. For example, you can change the model, dataset,
checkpoint, or other settings to meet your specific requirements. A full list of
configuration options can be found on the [configuration documentation
page](https://arctictraining.readthedocs.io/en/latest/config.html).

## Creating a New Trainer

If you want to create a new trainer, you can do so by subclassing the
``Trainer`` or ``SFTTrainer`` classes and implementing the necessary
modifications. For example, you could create a new trainer from ``SFTTrainer``
that uses a different loss function:

```python
from arctic_training import SFTTrainer

class CustomTrainer(SFTTrainer):
   name = "my_custom_trainer"

   def loss(self, batch):
       # Custom loss function implementation
       return loss
```

This new trainer will be automatically registered with ArcticTraining when the
script containing the declaration of ``CustomTrainer`` is imported.  By default,
ArcticTraining looks for a ``train.py`` in the current working directory to find
custom trainers. You can also specify a custom path to the trainers with the
``code`` field in your training recipe:

```yaml
type: my_custom_trainer
code: path/to/custom_trainers.py
model:
 name_or_path: meta-llama/Llama-3.1-8B-Instruct
data:
 sources:
   - HuggingFaceH4/ultrachat_200k
```
