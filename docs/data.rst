.. _data:

============
Data Factory
============

Data Source
-----------

The Data Source is responsible for loading the raw data used in the training
pipeline. A Data Source can be created by inheriting from the
:class:`~arctic_training.data.source.DataSource` class and implementing the
:meth:`~arctic_training.data.source.DataSource.load` method.

.. autoclass:: arctic_training.data.source.DataSource

Attributes
^^^^^^^^^^

To define a custom data source, you must subclass the DataSource and define the
:attr:`~.DataSource.name` attribute and give a type hint for the
:attr:`~.DataSource.config` attribute.

Methods
^^^^^^^

To define a custom data source, you must implement the
:meth:`~.DataSource.load_fn`. This method should return a HuggingFace Dataset
object.

Data Factory
------------

The Data Factory is responsible for creating the training and evaluation
datasets used in the training pipeline.

.. autoclass:: arctic_training.data.factory.DataFactory

Attributes
^^^^^^^^^^

To define a custom data factory, you must subclass the DataFactory, define the
:attr:`~.DataFactory.name` attribute, and give a type hint for the
:attr:`~.DataFactory.config` attribute.

Properties
^^^^^^^^^^

The Data Factory class provides several properties that can be used to access
information about the state of the Trainer, Tokenizer, and distributed
environment at runtime. These include :attr:`~.DataFactory.trainer`,
:attr:`~.DataFactory.tokenizer`, :attr:`~.DataFactory.micro_batch_size`,
:attr:`~.DataFactory.global_rank`, and :attr:`~.DataFactory.world_size`.

Methods
^^^^^^^

To define a custom data factory, you must implement the
:meth:`~.DataFactory.process` method.  Additionally, you can override the
:meth:`~.DataFactory.load`, :meth:`~.DataFactory.split_data`, and
:meth:`~.DataFactory.create_dataloader` methods to change default behaviors.

SFTDataFactory
--------------

To help get started with creating custom trainers and data factories,
ArcticTraining includes a Supervised Fine-Tuning (SFT) trainer (described in
:ref:`Trainer`). We also include here an example of how to build a data factory
from the base building blocks for use with the SFTTrainer. The SFTDataFactory
can be used with the SFTTrainer or your own custom trainer. It can also be
extended to fit other use cases.

To create the SFTDataFactory, we subclass the DataFactory and first define the
:meth:`~.DataFactory.process` method to tokenize the loaded datasets:

.. literalinclude:: ../arctic_training/data/sft_factory.py
   :pyobject: SFTDataFactory.process

Next we override the :meth:`~.DataFactory.create_dataloader` method to add a custom Data Collator:

.. literalinclude:: ../arctic_training/data/sft_factory.py
   :pyobject: SFTDataFactory.create_dataloader

Finally, we define two post-load callbacks that filter the any data source datasets
based on a maximum desired length and then pack the data:

.. literalinclude:: ../arctic_training/data/sft_factory.py
   :pyobject: filter_dataset_length

.. literalinclude:: ../arctic_training/data/sft_factory.py
   :pyobject: pack_dataset

These callback functions are added to SFTDataFactory by adding to the `callback`
attribute and they are run on the concatenated datasets returned from the
:meth:`~.DataFactory.load` method.

.. code-block:: python

    from arctic_training import logger
    class SFTDataFactory(DataFactory):
        name = "sft"
        config: SFTDataConfig
        callbacks = [
            ("post-load", filter_dataset_length),
            ("post-load", pack_dataset)
        ]
