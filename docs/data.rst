.. _data:

====
Data
====

Data Source
-----------

The Data Source is responsible for loading the raw data used in the training
pipeline. A Data Source can be created by inheriting from the
:class:`~arctic_training.data.source.DataSource` class and implementing the
:meth:`~arctic_training.data.source.DataSource.load_fn` method.

.. autoclass:: arctic_training.data.source.DataSource

Attributes
^^^^^^^^^^

To define a custom data source, you must subclass the DataSource and define the
following attributes: :attr:`~.DataSource.name` and
:attr:`~.DataSource.data_factory_type`.

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

To define a custom data factory, you must subclass the DataFactory and define
two attributes: :attr:`~.DataFactory.name` and
:attr:`~.DataFactory.config_type`.

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
:meth:`~.DataFactory.tokenizer_fn` and :meth:`~.DataFactory.collate_fn` methods.
Additionally, you can override the :meth:`~.DataFactory.modify_dataset` method
to apply any transformations to the dataset before it is returned.

SFTDataFactory
--------------

To help get started with creating custom trainers and data factories,
ArcticTraining includes a Supervised Fine-Tuning (SFT) trainer (described in
:ref:`Trainer`). We also include here an example of how to build a data factory
from the base building blocks for use with the SFTTrainer. The SFTDataFactory
can be used with the SFTTrainer or your own custom trainer. It can also be
extended to fit other use cases.

To create the SFTDataFactory, we subclass the DataFactory and first define the
:meth:`~.DataFactory.tokenizer_fn` method:

.. literalinclude:: ../arctic_training/data/sft_factory.py
   :pyobject: SFTDataFactory.tokenize_fn

Next we define a Data Collator class for the torch DataLoader:

.. literalinclude:: ../arctic_training/data/sft_factory.py
   :pyobject: DataCollatorForCausalLM

And we return define the :meth:`~.DataFactory.collate_fn` method where we return this object:

.. literalinclude:: ../arctic_training/data/sft_factory.py
   :pyobject: SFTDataFactory.collate_fn

Finally, we define a DataSource object that can be used with the SFTDataFactory.
Several data source examples are included in ArcticTraining. Here we define one
to load the HuggingFaceH4/ultrachat_200k dataset:

.. literalinclude:: ../arctic_training/data/sft_source.py
   :pyobject: UltraChat200K
