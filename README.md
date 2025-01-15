[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/snowflakedb/ArcticTraining/blob/main/LICENSE)

<h3 align="center">
| <a href="https://arctictraining.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="#"><b>Blog</b></a> | <a href="#"><b>Discource</b></a> |
</h3>

# ArcticTraining: Simplifying and Accelerating Post-Training for LLMs

ArcticTraining is a framework designed to simplify and accelerate the post-training process for large language models (LLMs). It addresses challenges in current frameworks, such as limited support for rapid prototyping and the lack of native data generation tools, by offering modular trainer designs, simplified code structures, and integrated pipelines for creating and cleaning synthetic data. These features enable users to enhance LLM capabilities, like code generation and complex reasoning, with greater efficiency and flexibility.

# Quickstart

To get started training a model with ArcticTraining, follow the steps below:

1. Clone the ArcticTraining repository and navigate to the root directory:

```bash
git clone https://github.com/snowflakedb/ArcticTraining.git
cd ArcticTraining
```

2. Install the ArcticTraining package and its dependencies:

```bash
pip install -e .
```

3. Create a training recipe YAML that uses the built-in Supervised Fine-Tuning (SFT) trainer:

```yaml
name: sft
micro_batch_size: 2
model:
  name_or_path: NousResearch/Meta-Llama-3.1-8B-Instruct
data:
  sources:
    - HuggingFaceH4/ultrachat_200k
checkpoint:
  - type: huggingface
    save_end_of_training: true
    output_dir: ./fine-tuned-model
```

4. Run the training recipe with the ArcticTraining CLI (see below). This will use the `DeepSpeed` launcher behind the scenes, you can pass any compatible DeepSpeed launcher arguments to the ArcticTraining CLI (e.g., --num_nodes, --num_gpus).

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
from arctic_training.trainers import SFTTrainer, register

@register
class CustomTrainer(SFTTrainer):
   name = "my_custom_trainer"

   def loss(self, batch):
       # Custom loss function implementation
       return loss
```

Remember to register this new trainer using the ``@register`` decorator so that
it can be used in training recipes. By default, ArcticTraining looks for a
``train.py`` in the current working directory to find custom trainers. You can
also specify a custom path to the trainers with the ``code`` field in your
training recipe:

```yaml
name: my_custom_trainer
code: path/to/custom_trainers.py
model:
 name_or_path: NousResearch/Meta-Llama-3.1-8B-Instruct
data:
 sources:
   - HuggingFaceH4/ultrachat_200k
```
