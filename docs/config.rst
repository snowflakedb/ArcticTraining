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
