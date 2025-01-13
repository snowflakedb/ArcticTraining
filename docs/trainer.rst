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

There are several attributes that must be defined in the Trainer class to create
a new custom trainer. These attributes include: :attr:`.name`,
:attr:`.config_type`, :attr:`.data_factory_type`, :attr:`.model_factory_type`,
:attr:`.checkpoint_engine_type`, :attr:`.optimizer_factory_type`,
:attr:`.scheduler_factory_type`, and :attr:`.tokenizer_factory_type`.

These attributes are used when registering new custom trainers with
ArcticTraining and to validate training recipes that use the trainer.

Properties
----------

The Trainer class provides several properties that can be used to access
information about the state of the trainer at runtime. These include
:attr:`.epochs`, :attr:`.train_batches`, :attr:`.device`,
:attr:`.training_horizon`, and :attr:`.warmup_steps`.

Properties should typically not be set by custom trainers, but can be used by
other custom classes, like new checkpoint engines or model factories, to access
information about the training process.

Methods
-------

The Trainer class has several methods that divide the training loop into
segments. At minimum, a new trainer must specify the :meth:`.loss` method.
However any of the :meth:`.train`, :meth:`.epoch`, :meth:`.step`, or
:meth:`.checkpoint` methods can be overridden to customize the training process.
