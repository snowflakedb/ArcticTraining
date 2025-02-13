.. _optimizer:

=================
Optimizer Factory
=================

The :class:`~.OptimizerFactory` is responsible for generating the optimizer used
in training from the model created with the
:class:`arctic_training.model.factory.ModelFactory`. An Optimizer Factory can be
created by inheriting from the :class:`~.OptimizerFactory` class and
implementing the :meth:`~.OptimizerFactory.create_optimizer` method.

.. autoclass:: arctic_training.optimizer.factory.OptimizerFactory

Attributes
----------

Similar to other Factory classes in ArcticTraining, the
:class:`~.OptimizerFactory` class must have a :attr:`~.OptimizerFactory.name`
attribute that is used to identify the factory when registering it with
ArcticTraining and a :attr:`~.OptimizerFactory.config` attribute type hint that
is used to validate the config object passed to the factory.

Properties
----------

:class:`~.OptimizerFactory` has several attributes that can be used to access
information about the trainer and distributed state at runtime, including
:attr:`~.OptimzerFactory.device`, :attr:`~.OptimizerFactory.trainer`,
:attr:`~.OptimizerFactory.model`, :attr:`~.OptimizerFactory.world_size`, and
:attr:`~.OptimizerFactory.global_rank`.

Methods
-------

The :class:`~.OptimizerFactory` has just one method that must be defined:
:meth:`~.OptimizerFactory.create_optimizer`. Given a model and optimizer config,
the method should return the optimizer.

Adam Optimizer Factory
-----------------------

As an example of how to create a new :class:`~.OptimizerFactory`, we provide the
:class:`arctic_training.optimizer.factory.FusedAdamOptimizerFactory` which
returns the `FusedAdam` optimizer from the `DeepSpeed library
<https://github.com/Microsoft/DeepSpeed>`_.
