.. _scheduler:

=================
Scheduler Factory
=================

The :class:`~.SchedulerFactory` is responsible for generating the scheduler used
in training from the optimizer created by the
:class:`arctic_training.optimizer.factory.OptimizerFactory`. A Scheduler Factory
can be created by inheriting from the :class:`~.SchedulerFactory` class and
implementing the :meth:`~.SchedulerFactory.create_scheduler` method.

.. autoclass:: arctic_training.scheduler.factory.SchedulerFactory

Attributes
----------

Similar to other Factory classes in ArcticTraining, the SchedulerFactory class
must have a :attr:`~.SchedulerFactory.name` attribute that is used to identify
the factory when registering it with ArcticTraining and a
:attr:`~.SchedulerFactory.config` attribute type hint that is used to validate
the config object passed to the factory.

Properties
----------

A SchedulerFactory has several attributes that can be used to access information
about the trainer and distributed state at runtime, including
:attr:`~.SchedulerFactory.device`, :attr:`~.SchedulerFactory.trainer`,
:attr:`~.SchedulerFactory.optimizer`, :attr:`~.SchedulerFactory.world_size`, and
:attr:`~.SchedulerFactory.global_rank`.

Methods
-------

SchedulerFactories have just one method that must be defined:
:meth:`~.SchedulerFactory.create_scheduler`. This method should return the
scheduler object created using the optimizer object passed to the factory.

Huggingface Scheduler Factory
-----------------------------

A custom scheduler factory can be created from the :class:`~.SchedulerFactory`,
but ArcticTraining comes with a
:class:`~.arctic_training.scheduler.hf_factory.HFSchedulerFactory`
implementation that can be used out of the box.
