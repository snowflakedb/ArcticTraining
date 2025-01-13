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
