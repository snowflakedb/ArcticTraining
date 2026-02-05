# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import transformers

from arctic_training import snowflake_connection
from arctic_training.checkpoint.engine import CheckpointEngine
from arctic_training.config.checkpoint import CheckpointConfig
from arctic_training.logging import logger


class SnowflakeCheckpointConfig(CheckpointConfig):
    """Configuration for Snowflake Model Registry checkpoint engine."""

    model_name: str
    """ Name for the model in Snowflake Model Registry. """


class SnowflakeCheckpointEngine(CheckpointEngine):
    """Checkpoint engine that logs models to Snowflake Model Registry.

    This engine exports trained models to Snowflake Model Registry using the
    transformers.pipeline approach. The model is logged with a version name
    based on the global training step.

    Note: This engine does not support PEFT/adapter models. Only full models
    can be logged to Snowflake Model Registry.
    """

    name = "snowflake"
    config: SnowflakeCheckpointConfig

    def _infer_pipeline_task(self, model: Any) -> str:
        """Infer the pipeline task from the model architecture.

        Args:
            model: The HuggingFace model to infer the task from.

        Returns:
            The pipeline task string (e.g., "text-generation").
        """
        model_class_name = model.__class__.__name__

        # Map model architecture to pipeline task
        if "ForCausalLM" in model_class_name:
            return "text-generation"
        elif "ForSeq2SeqLM" in model_class_name:
            return "text2text-generation"
        elif "ForSequenceClassification" in model_class_name:
            return "text-classification"
        elif "ForTokenClassification" in model_class_name:
            return "token-classification"
        elif "ForQuestionAnswering" in model_class_name:
            return "question-answering"
        elif "ForMaskedLM" in model_class_name:
            return "fill-mask"
        else:
            # Default to text-generation for this training framework
            logger.warning(
                f"Could not infer pipeline task from model class {model_class_name}. Defaulting to 'text-generation'."
            )
            return "text-generation"

    def save(self, model: Any) -> None:
        """Save the model to Snowflake Model Registry.

        Args:
            model: The DeepSpeed model engine containing the trained model.

        Raises:
            ValueError: If PEFT/adapter model is used (not supported by Snowflake).
        """
        # Check for PEFT config - Snowflake does not support adapter models
        if self.trainer.config.model.peft_config is not None:
            raise ValueError(
                "Snowflake Model Registry does not support PEFT/adapter models. "
                "Only full models can be logged. Please use a different checkpoint "
                "engine (e.g., 'huggingface' or 'deepspeed') for PEFT models."
            )

        # Only rank 0 logs to the registry
        if self.global_rank != 0:
            return

        # Extract the underlying HuggingFace model from DeepSpeed engine
        hf_model = model.module if hasattr(model, "module") else model

        # Infer the pipeline task from model architecture
        task = self._infer_pipeline_task(hf_model)

        # Get the tokenizer
        tokenizer = self.trainer.tokenizer

        # Create a transformers pipeline
        pipeline = transformers.pipeline(
            task=task,
            model=hf_model,
            tokenizer=tokenizer,
        )

        # Get Snowflake session
        session = snowflake_connection.get_default_snowflake_session()

        # Import Snowflake ML registry
        from snowflake.ml.registry import Registry

        # Create registry and log the model
        registry = Registry(session=session)
        version_name = f"global_step_{self.trainer.global_step}"

        logger.info(
            "Logging model to Snowflake Model Registry: "
            f"model_name={self.config.model_name}, version_name={version_name}"
        )

        registry.log_model(
            model=pipeline,
            model_name=self.config.model_name,
            version_name=version_name,
        )

        logger.info(f"Successfully logged model to Snowflake Model Registry: {self.config.model_name}/{version_name}")

    @property
    def latest_checkpoint_exists(self) -> bool:
        """Check if a checkpoint exists in the registry.

        Not implemented for Snowflake checkpoint engine as it is export-only.
        """
        raise NotImplementedError(
            "SnowflakeCheckpointEngine is export-only and does not support checking for existing checkpoints."
        )

    def load(self, model: Any) -> None:
        """Load a model from Snowflake Model Registry.

        Not implemented for Snowflake checkpoint engine as it is export-only.
        """
        raise NotImplementedError(
            "SnowflakeCheckpointEngine is export-only and does not support "
            "loading models. Use the Snowflake Model Registry API directly "
            "to load models for inference."
        )
