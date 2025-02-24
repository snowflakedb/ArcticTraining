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
