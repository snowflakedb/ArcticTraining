.. _trainer:

=======
Trainer
=======

ArcticTraining provides a flexible and extensible training framework that allows
you to customize and create your own training workflows. At the core of this
framework is the Trainer class, which orchestrates the training process by
managing the model, optimizer, data loader, and other components.

The Trainer class is designed to be modular and extensible, allowing you to
quickly swap in and out different building blocks to experiment with different
training strategies. Here, we'll walk through the key features of the
Trainer class and show you how to create your own custom trainers.

.. autoclass:: arctic_training.trainer.trainer::Trainer

Attributes
----------

.. _trainer-attributes:

There are several attributes that must be defined in the Trainer class to create
a new custom trainer. These attributes include: :attr:`~.Trainer.name`,
:attr:`~.Trainer.config_type`, :attr:`~.Trainer.data_factory_type`,
:attr:`~.Trainer.model_factory_type`, :attr:`~.Trainer.checkpoint_engine_type`,
:attr:`~.Trainer.optimizer_factory_type`,
:attr:`~.Trainer.scheduler_factory_type`, and
:attr:`~.Trainer.tokenizer_factory_type`.

These attributes are used when registering new custom trainers with
ArcticTraining and to validate training recipes that use the trainer.

Properties
----------

The Trainer class provides several properties that can be used to access
information about the state of the trainer at runtime. These include
:attr:`~.Trainer.epochs`, :attr:`~.Trainer.train_batches`,
:attr:`~.Trainer.device`, :attr:`~.Trainer.training_horizon`, and
:attr:`~.Trainer.warmup_steps`.

Properties should typically not be set by custom trainers, but can be used by
other custom classes, like new checkpoint engines or model factories, to access
information about the training process.

Methods
-------

The Trainer class has several methods that divide the training loop into
segments. At minimum, a new trainer must specify the :meth:`~.Trainer.loss`
method.  However any of the :meth:`~.Trainer.train`, :meth:`~.Trainer.epoch`,
:meth:`~.Trainer.step`, or :meth:`~.Trainer.checkpoint` methods can be
overridden to customize the training process.

Train
^^^^^

.. literalinclude:: ../arctic_training/trainer/trainer.py
   :pyobject: Trainer.train

Epoch
^^^^^

.. literalinclude:: ../arctic_training/trainer/trainer.py
   :pyobject: Trainer.epoch

Step
^^^^

.. literalinclude:: ../arctic_training/trainer/trainer.py
   :pyobject: Trainer.step

Checkpoint
^^^^^^^^^^

.. literalinclude:: ../arctic_training/trainer/trainer.py
   :pyobject: Trainer.checkpoint

Supervised Fine-Tuning (SFT) Trainer
-------------------------------------

To help you get started with creating custom trainers, ArcticTraining includes a
Supervised Fine-Tuning (SFT) trainer that demonstrates how to build a training
pipeline from the base building blocks. The SFT trainer can in turn be used as a
starting point and extended for creating your own custom trainers.

To create the SFT trainer, we subclass the Trainer and override the
:meth:`~.Trainer.loss` method. We also define the necessary components described
in :ref:`Trainer Attributes<trainer-attributes>`. We use a custom data factory,
SFTDataFactory, which we describe in greater detail in the :ref:`Data
Factory<data>` section. The remainder of the attributes use the base building
blocks from ArcticTraining. For example the model factory defaults to the
HFModelFactory (because it is listed first in the model_factory_type attribute),
but this trainer can work with either `HFModelFactory` or `LigerModelFactory`.

.. literalinclude:: ../arctic_training/trainer/sft_trainer.py
   :pyobject: SFTTrainer
