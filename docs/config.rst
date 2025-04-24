.. _config:

=============
Configuration
=============

The main input to the ArcticTraining CLI is a YAML configuration file that
defines files for the :class:`~arctic_training.config.trainer.TrainerConfig`
class. This is a Pydantic configuration model that also contains the
sub-configurations for data, model, etc.

.. autopydantic_model:: arctic_training.config.trainer.TrainerConfig

.. autopydantic_model:: arctic_training.config.checkpoint.CheckpointConfig

.. autopydantic_model:: arctic_training.config.data.DataConfig

.. autopydantic_model:: arctic_training.config.logger.LoggerConfig

.. autopydantic_model:: arctic_training.config.model.ModelConfig

.. autopydantic_model:: arctic_training.config.optimizer.OptimizerConfig

.. autopydantic_model:: arctic_training.config.scheduler.SchedulerConfig

.. autopydantic_model:: arctic_training.config.tokenizer.TokenizerConfig

.. autopydantic_model:: arctic_training.config.wandb.WandBConfig

Numerical Formatting
--------------------

When specifying numerical values in the configuration file, you can use
human-friendly strings to represent very large or very small numbers. The
following formats are supported:

- ``X%``: This format represents a percentage. For example, ``50%`` is equivalent to ``0.5``.
- ``XeY``: This format represents a number in scientific notation. For example,
  ``1e-6`` is equivalent to ``0.000001``.
- ``X^Y``: This format represents a number raised to a power. For example,
  ``2^20`` is equivalent to ``1048576``.
- ``XK``: This format represents a number in thousands (base 10). For example,
  ``1K`` is equivalent to ``1000``. Similarly you can use ``M`` for millions,
  ``B`` for billions, and ``T`` for trillions.
- ``1Ki``: This format represents a number in kibibytes (base 2). For example,
  ``1Ki`` is equivalent to ``1024``. Similarly you can use ``Mi`` for mebibytes,
  ``Gi`` for gibibytes, and ``Ti`` for tebibytes.
