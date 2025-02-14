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

Creating a custom trainer starts with Inheriting from the base
:class:`~.Trainer` class and defining the :attr:`~.Trainer.name` attribute. The
name attribute is used to identify the trainer when registering it with
ArcticTraining. Additionally, you can define custom types for
:attr:`~.Trainer.config`, :attr:`~.Trainer.data_factory`,
:attr:`~.Trainer.model_factory`, :attr:`~.Trainer.checkpoint_engine`,
:attr:`~.Trainer.optimizer_factory`, :attr:`~.Trainer.scheduler_factory`, and
:attr:`~.Trainer.tokenizer_factory` to specify the default factories for each
component.

Specify the type hint for these attributes tells ArcticTraining which building
blocks are compatible with your custom trainer. You may define multiple
compatible building blocks by using `typing.Union` in the type hint. When
multiple types are specified for one of these attributes, the first is used as a
default in the case where `type` is not specified in the input config.

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
HFModelFactory (because it is listed first in the ``model_factory`` attribute
type hint), but this trainer can work with either `HFModelFactory` or
`LigerModelFactory`.

.. literalinclude:: ../arctic_training/trainer/sft_trainer.py
   :pyobject: SFTTrainer
