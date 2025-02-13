.. _checkpoint:

=================
Checkpoint Engine
=================

The checkpoint engine in ArcticTraining allows you to save the model in the
middle of training and/or after training has completed. Checkpoint engines can
be implemented from the base :class:`~.CheckpointEngine` class by implementing
the :meth:`~.CheckpointEngine.load` and :meth:`~.CheckpointEngine.save` methods.

.. autoclass:: arctic_training.checkpoint.engine.CheckpointEngine

Attributes
----------

Similar to the ``*Factory`` classes of ArcticTraining, the CheckpointEngine
class requires only the :attr:`~.CheckpointEngine.name` be defined and the
:attr:`~.CheckpointEngine.config` attribute type hint. The ``name`` attribute is
used to identify the engine when registering it with ArcticTraining and the
``config`` attribute type hint is used to validate the config object passed to
the engine.

Properties
----------

A CheckpointEngine has several attributes that can be used to access information
about the trainer and distributed state at runtime, including
:attr:`~.CheckpointEngine.device`, :attr:`~.CheckpointEngine.trainer`,
:attr:`~.CheckpointEngine.world_size`, and
:attr:`~.CheckpointEngine.global_rank`. Additionally, the base
:class:`~.CheckpointEngine` includes some unique properties that are helpful for
building new checkpoint engines, such as
:attr:`~.CheckpointEngine.do_checkpoint` (which checks if a checkpoint should be
saved) and :attr:`~.CheckpointEngine.checkpoint_dir` (which specifies the
directory where the checkpoint should be saved).

Methods
-------

CheckpointEngines have just two methods that must be defined:
:meth:`~.CheckpointEngine.load` and :meth:`~.CheckpointEngine.save`.  The
:meth:`~.CheckpointEngine.load` method should accept an intialized model and
load the model weights from an existing checkpoint.  The
:meth:`~.CheckpointEngine.save` method should save the model to a checkpoint
directory.

HuggingFace and DeepSpeed Checkpoint Engines
--------------------------------------------

While a custom checkpoint engine can be created from the
:class:`~.CheckpointEngine`, Arctic Training includes two CheckpointEngine
implementations that can be used out of the box: :class:`~.HFCheckpointEngine`
and :class:`~.DSCheckpointEngine`.

The :class:`~.HFCheckpointEngine` will save the model in a HuggingFace Hub style
using ``safetensor`` outputs. These checkpoints do no save the optimizer state
and thus are not compatible with resuming training from a checkpoint. As a
result, the :meth:`~.HFCheckpointEngine.load` method will raise an error if we
attempt to load a model from this style of checkpoint. This checkpoint engine is
useful for saving the model at the end of training for use with inference
libraries like `vLLM <https://github.com/vllm-project/vllm>`_.

The :class:`~.DeepSpeedCheckpointEngine` uses the checkpoint capabilities from
the DeepSpeed library. These style of checkpoints save the optimizer state and
can be used to resume training. This checkpoint engine is useful for saving
training progress during the training loop.
