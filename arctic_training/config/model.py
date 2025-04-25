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

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

import peft
from pydantic import field_validator

from arctic_training.config.base import BaseConfig
from arctic_training.config.enums import DType
from arctic_training.registry import get_registered_model_factory

if TYPE_CHECKING:
    from arctic_training.model.factory import ModelFactory


class ModelConfig(BaseConfig):
    type: str = ""
    """ Model factory type. """

    name_or_path: Union[str, Path]
    """ Model name (as described in Hugging Face model hub) or local path to model checkpoint. """

    dtype: DType = DType.BF16
    """ Data type for model weights. """

    save_name: Optional[str] = None
    """ Name to use when saving the model. """

    attn_implementation: str = "sdpa"
    """ Attention implementation to use. """

    disable_activation_checkpoint: bool = False
    """ Disable the use of activation checkpointing. """

    peft_config: Optional[Dict] = None
    """ Configuration for Parameter Efficient Fine Tuning. """

    @property
    def factory(self) -> Type["ModelFactory"]:
        return get_registered_model_factory(name=self.type)

    @property
    def peft_config_obj(self) -> peft.PeftConfig:
        if self.peft_config is None:
            raise ValueError("No PEFT config specified.")
        peft_config_cls = getattr(peft, f"{self.peft_config['peft_type']}Config")
        return peft_config_cls(**self.peft_config)

    @field_validator("peft_config", mode="before")
    @classmethod
    def validate_peft_config_type(cls, value: Optional[Dict]) -> Optional[Dict]:
        if value is not None:
            if "peft_type" not in value:
                raise ValueError("No 'peft_type' specified in PEFT config.")
            peft_type = value["peft_type"]

            valid_peft_types = [key.removesuffix("Config") for key in peft.__dict__.keys() if key.endswith("Config")]
            if peft_type not in valid_peft_types:
                raise ValueError(f"PEFT type {peft_type} config not found. Valid PEFT types are: {valid_peft_types}")

        return value

    @field_validator("attn_implementation", mode="after")
    def validate_attn_implementation(cls, value: str) -> str:
        if value == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ValueError(
                    "flash_attention_2 requires the flash_attn package. Install with"
                    " `pip install flash_attn`. Please refer to documentation at"
                    " https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2"
                )
        return value
