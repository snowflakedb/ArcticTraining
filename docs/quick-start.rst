
.. _quickstart:

===========
Quick Start
===========

To get started training a model with ArcticTraining, follow the steps below:

1. Clone the ArcticTraining repository and navigate to the root directory:

   .. code-block:: bash

      git clone https://github.com/snowflakedb/ArcticTraining.git && cd ArcticTraining

2. Install the ArcticTraining package and its dependencies:

   .. code-block:: bash

      pip install -e .

3. Create a training recipe YAML that uses the built-in Supervised Fine-Tuning (SFT) trainer:

   .. code-block:: yaml

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

4. Run the training recipe with the ArcticTraining CLI:

   .. code-block:: bash

      arctic_training path/to/recipe.yaml

Customize Training
------------------

To customize the training workflow, you can modify the training recipe YAML we
created in step 3 above. For example, you can change the model, dataset,
checkpoint, or other settings to meet your specific requirements. A full list of
configuration options can be found on the :ref:`config page <config>`.

Creating a New Trainer
^^^^^^^^^^^^^^^^^^^^^^

If you want to create a new trainer, you can do so by subclassing the
``Trainer`` or ``SFTTrainer`` classes and implementing the necessary
modifications. For example, you could create a new trainer from ``SFTTrainer``
that uses a different loss function:

.. code-block:: python

   from arctic_training import SFTTrainer

   class CustomTrainer(SFTTrainer):
       name = "my_custom_trainer"

       def loss(self, batch):
           # Custom loss function implementation
           return loss

This new trainer will be automatically registered with ArcticTraining when the
script containing the declaration of ``CustomTrainer`` is imported. By default,
ArcticTraining looks for a ``train.py`` in the current working directory to find
custom trainers. You can also specify a custom path to the trainers with the
``code`` field in your training recipe:

.. code-block:: yaml

   type: my_custom_trainer
   code: path/to/custom_trainers.py
   model:
     name_or_path: meta-llama/Llama-3.1-8B-Instruct
   data:
     sources:
       - HuggingFaceH4/ultrachat_200k
