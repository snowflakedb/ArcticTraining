.. _model:

=============
Model Factory
=============

The Model Factory is responsible for generating the model used in training from
the ModelConfig. A Model Factory can be created by inheriting from the
:class:`~.ModelFactory` class and implementing the
:meth:`~.ModelFactory.create_config` and :meth:`~.ModelFactory.create_model`
methods.

.. autoclass:: arctic_training.model.factory.ModelFactory

Attributes
----------

Similar to other Factory classes in ArcticTraining, the ModelFactory class must
have a :attr:`~.ModelFactory.name` attribute that is used to identify the
factory when registering it with ArcticTraining and a
:attr:`~.ModelFactory.config` attribute type hint that is used to validate the
config object passed to the factory.

Properties
----------

A ModelFactory has several attributes that can be used to access information
about the trainer and distributed state at runtime, including
:attr:`~.ModelFactory.device`, :attr:`~.ModelFactory.trainer`,
:attr:`~.ModelFactory.world_size`, and :attr:`~.ModelFactory.global_rank`.

Methods
-------

ModelFactories have just two methods that must be defined:
:meth:`~.ModelFactory.create_config` and :meth:`~.ModelFactory.create_model`.
The :meth:`~.ModelFactory.create_config` method should return a config object
that can be used to generate the desired model and the
:meth:`~.ModelFactory.create_model` method should return the model object
created using the generated config.

HuggingFace Style Factories
---------------------------

A custom model factory can be created from the :class:`~.ModelFactory` building
block, but ArcticTraining also comes with two ModelFactory implementations that
can be used out of the box: :class:`~.HFModelFactory` and
:class:`~.LigerModelFactory`. Each of these will load models from HuggingFace
Hub given a path to a local repo or the model name. The
:class:`~.LigerModelFactory` extends :class:`~.HFModelFactory` and adds support
for using optimizations in the `Liger-Kernel
library <https://github.com/linkedin/Liger-Kernel>`_.
