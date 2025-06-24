.. _usage:

=====
Usage
=====

After :ref:`installation <install>`, you can use ArcticTraining to train
your models using a simple YAML recipe or a Python script. Here we provide an
overview of how to use each.

ArcticTraining CLI
------------------

The ArcticTraining CLI is the easiest way to train your models using
ArcticTraining and supports the use of custom trainers, data, etc. to meet your
specific requirements. To train a model using the ArcticTraining CLI, follow
these steps:

1. Create a training recipe YAML file with the necessary configuration options.
   For example, you can create a recipe to train a model using the
   `meta-llama/Llama-3.1-8B-Instruct
   <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>`_ model and
   the `HuggingFaceH4/ultrachat_200k
   <https://huggingface.co/HuggingFaceH4/ultrachat_200k>`_ dataset:

   .. code-block:: yaml

      model:
        name_or_path: meta-llama/Llama-3.1-8B-Instruct
      data:
        sources:
          - HuggingFaceH4/ultrachat_200k
      checkpoint:
        - type: huggingface
          save_end_of_training: true
          output_dir: ./fine-tuned-model

2. Optionally create a custom trainer by subclassing the ``Trainer`` or
   ``SFTTrainer`` classes and implementing the necessary modifications. For
   example, you could create a new trainer from ``SFTTrainer`` that uses a
   different loss function:

   .. code-block:: python

      from arctic_training import SFTTrainer

      class CustomTrainer(SFTTrainer):
          name = "my_custom_trainer"

          def loss(self, batch):
              # Custom loss function implementation
              return loss

   This new trainer will be automatically registered with ArcticTraining when
   the script containing the declaration of ``CustomTrainer`` is imported. By
   default, ArcticTraining looks for a ``train.py`` in the directory where the
   YAML training recipe is located to find custom trainers. You can also specify
   a custom path to the trainers with the ``code`` field in your training
   recipe:

   .. code-block:: yaml

      type: my_custom_trainer
      code: path/to/custom_trainers.py
      model:
        name_or_path: meta-llama/Llama-3.1-8B-Instruct
      data:
        sources:
          - HuggingFaceH4/ultrachat_200k

   You may also wish to create a new model factory, data factory, etc. to
   accompany your new trainer. This can also be done in the same python script
   and these classes will automatically be registered as well:

   .. code-block:: python

      from arctic_training import HFModelFactory, SFTTrainer

      class CustomModelFactory(HFModelFactory):
          name = "my_custom_model_factory"

          def create_model(self, config):
              # Custom model implementation
              return model

      class CustomTrainer(SFTTrainer):
          name = "my_custom_trainer"
          model_factory: CustomModelFactory

          def loss(self, batch):
              # Custom loss function implementation
              return loss

3. Run the training recipe with the ArcticTraining CLI:

   .. code-block:: bash

      arctic_training path/to/recipe.yaml

   Under the hood our CLI will load the recipe, instantiate the trainer, model,
   etc. and start training.

   Our CLI launcher uses the DeepSpeed launcher to create a distributed training
   environment. You can pass any DeepSpeed arguments after the training recipe
   path. For example, to train on 4 GPUs, you can run:

    .. code-block:: bash

        arctic_training path/to/recipe.yaml --num_gpus 4

Python API
----------

ArcticTraining also provides a Python API that can be used to setup trainer and
train your model. Here we show the same example as above but using the Python
API:

.. code-block:: python

    from arctic_training import HFModelFactory, SFTTrainer, get_config

    class CustomModelFactory(HFModelFactory):
        name = "my_custom_model_factory"

        def create_model(self, config):
            # Custom model implementation
            return model

    class CustomTrainer(SFTTrainer):
        name = "my_custom_trainer"
        model_factory: CustomModelFactory

        def loss(self, batch):
            # Custom loss function implementation
            return loss

    if __name__ == "__main__":
        config_dict = {
            "type": "my_custom_trainer",
            "model": {
                "name_or_path": "meta-llama/Llama-3.1-8B-Instruct"
            },
            "data": {
                "sources": ["HuggingFaceH4/ultrachat_200k"]
            }
            "checkpoint": [
                {
                    "type": "huggingface",
                    "save_end_of_training": True,
                    "output_dir": "./fine-tuned-model"
                }
            ]
        }

        config = get_config(config_dict)
        trainer = CustomTrainer(config)
        trainer.train()


Datasets
----------

How to use a dataset of your choice. Since there is no standard to how each dataset is defined it's not always easy to write a generic API that will work with any dataset.

SFT Datasets
============

While one could write a class for any new dataset, we have designed a flexible dataset type that should allow to remap many existing Instruct/SFT datasets to this dataset type:

The ``role_mapping`` dict indicates how to locate the role and content
within the dataset structure. We accept two types of inputs:

1. ``{role_name} : {column_name}``

2. ``{role_name} : {column_name.filter_field.filter_value}``

Additionally ``content_key`` can be used when a deep structure with
complex columns is used and the value name needs remapping, see example
5 below for such a use-case.

Examples:

1. Dataset structure:

.. code:: python

   {"user": "What is the capital of France?", "assistant": "The capital of France is Paris."}

Config:

.. code:: yaml

   data:
     sources:
       - type: huggingface_instruct
         name_or_path:  Josephgflowers/Finance-Instruct-500k
         split: train

See https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k

2. Dataset structure:

.. code:: python

   {"instruction": "What is the capital of France?", "demonstration": "The capital of France is Paris."}

Config:

.. code:: yaml

   data:
     sources:
       - type: huggingface_instruct
         name_or_path: HuggingFaceH4/helpful-instructions
         split: train
         sample_count: 1000
         role_mapping:
           user: instruction
           assistant: demonstration

See https://huggingface.co/datasets/HuggingFaceH4/helpful-instructions

3. Dataset structure:

.. code:: python

   {"messages": [{"role": "user", "content": "Hello world"}, {"role": "assistant", "content": "Hi there"}]}

Config:

.. code:: yaml

   data:
     sources:
       - type: huggingface_instruct
         name_or_path: HuggingFaceH4/ultrachat_200k
         split: train_sft
         role_mapping:
           user: messages.role.user
           assistant: messages.role.assistant

See https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k

4. Dataset structure:

.. code:: python

   {"conversations": [{"role": "human", "content": "Hello world"}, {"role": "agent", "content": "Hi there"}]}

Config:

.. code:: yaml

   data:
     sources:
       - type: huggingface_instruct
         name_or_path: /path/to/data
         role_mapping:
           user: conversations.role.human
           assistant: conversations.role.agent

5. Dataset structure:

.. code:: python

   {"conversations": [{"sender": "system", "message": "Hello world"}, {"sender": "user, "message": "Hi there"}]}

Config:

.. code:: yaml

   data:
     sources:
       - type: huggingface_instruct
         name_or_path: recursal/Europarl-Translation-Instruct
         split: full
         sample_count: 1000
         role_mapping:
           user: conversations.sender.system
           assistant: conversations.sender.user
           content_key: message

https://huggingface.co/datasets/recursal/Europarl-Translation-Instruct
additionally has the user/assistant roles reversed, so we can easily fix
it in our remapping.
