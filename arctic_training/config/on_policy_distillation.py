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

"""Configuration for On-Policy Distillation Trainer.

On-Policy Distillation trains a student model by having it generate its own
trajectories, then using a teacher model to provide per-token supervision via
reverse KL divergence. This contrasts with traditional (off-policy) distillation
where the teacher generates trajectories for the student to imitate.
"""

from typing import Dict
from typing import Union
from typing import cast

from pydantic import Field
from pydantic import ValidationInfo
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self

from arctic_training.config.model import ModelConfig
from arctic_training.config.trainer import TrainerConfig
from arctic_training.config.utils import HumanInt
from arctic_training.registry import get_registered_model_factory


class OnPolicyDistillationTrainerConfig(TrainerConfig):
    """Configuration for On-Policy Distillation Trainer.

    On-policy distillation trains the student on its own generated trajectories,
    with the teacher providing dense per-token feedback via reverse KL divergence.
    """

    teacher_model: ModelConfig
    """
    Configuration for the teacher model used in on-policy distillation.
    The teacher model provides per-token log probabilities for computing
    the reverse KL divergence loss against student-generated trajectories.
    """

    teacher_deepspeed: Dict = {}
    """
    DeepSpeed configuration for the teacher model. This is automatically
    computed based on the main model's DeepSpeed config and should not
    be provided by the user.
    """

    disable_teacher_dropout: bool = True
    """
    Whether to disable dropout in the teacher model during training.
    Recommended to keep True for stable distillation signal.
    """

    num_rollouts_per_prompt: int = Field(default=4, ge=1)
    """
    Number of trajectory samples to generate from the student per prompt.
    Higher values provide more diverse on-policy samples but increase compute.
    """

    max_new_tokens: HumanInt = Field(default=2048, ge=1)
    """
    Maximum number of new tokens to generate for each student trajectory.
    Should be set based on expected response length for the task.
    """

    generation_temperature: float = Field(default=1.0, gt=0.0)
    """
    Temperature for student trajectory generation.
    Higher values produce more diverse samples but may reduce quality.
    """

    generation_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    """
    Top-p (nucleus) sampling parameter for student generation.
    """

    generation_top_k: int = Field(default=0, ge=0)
    """
    Top-k sampling parameter for student generation. 0 means no top-k filtering.
    """

    beta: float = Field(default=1.0, gt=0.0)
    """
    Coefficient for the reverse KL divergence loss.
    Controls the strength of the distillation signal.
    """

    generation_batch_size: int = Field(default=0, ge=0)
    """
    Batch size for trajectory generation. If 0, uses micro_batch_size.
    May need to be smaller than micro_batch_size due to memory constraints
    during generation.
    """

    @field_validator("teacher_model", mode="before")
    @classmethod
    def init_teacher_model_config(cls, v: Union[Dict, ModelConfig], info: ValidationInfo) -> ModelConfig:
        """Initialize teacher model config from dict or ModelConfig."""
        subconfig = cls._get_subconfig_object(
            v=v,
            info=info,
            get_class_fn=get_registered_model_factory,
            attr_name="teacher_model_factory",
        )
        return cast(ModelConfig, subconfig)

    @model_validator(mode="after")
    def build_teacher_deepspeed_config(self) -> Self:
        """Build DeepSpeed config for teacher model."""
        if len(self.teacher_deepspeed) != 0:
            raise ValueError(
                "Teacher model DeepSpeed config is computed based on the main model "
                "DeepSpeed config and should not be passed by the user."
            )

        teacher_deepspeed = dict(
            train_batch_size=self.deepspeed["train_batch_size"],
            train_micro_batch_size_per_gpu=self.deepspeed["train_micro_batch_size_per_gpu"],
            steps_per_print=self.deepspeed["steps_per_print"],
            zero_optimization=dict(
                stage=3 if self.deepspeed["zero_optimization"]["stage"] == 3 else 0,
                stage3_param_persistence_threshold=1e4,
                memory_efficient_linear=False,
            ),
            bfloat16=dict(enabled=True),
            gradient_clipping=1.0,
            prescale_gradients=False,
            wall_clock_breakdown=False,
        )
        self.teacher_deepspeed = teacher_deepspeed
        return self

    @model_validator(mode="after")
    def set_generation_batch_size(self) -> Self:
        """Set generation batch size to micro_batch_size if not specified."""
        if self.generation_batch_size == 0:
            self.generation_batch_size = self.micro_batch_size
        return self
