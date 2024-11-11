from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import model_validator
from typing_extensions import Self

from arctic_training.model import get_optimizer_grouped_parameters

from .base import BaseConfig
from .enums import DType


class ModelConfig(BaseConfig):
    name_or_path: Union[str, Path]
    dtype: DType = DType.BF16
    save_name: Optional[str] = None
    use_liger_kernel: bool = True
    disable_flash_attn: bool = False
    disable_activation_checkpoint: bool = False
    adjustments: List[Callable] = []
    create_optimizer_grouped_params: Callable = get_optimizer_grouped_parameters
    peft_config: Dict[str, Any] = {}

    @property
    def attn_implementation(self) -> str:
        if self.disable_flash_attn:
            return "eager"
        return "flash_attention_2"

    @model_validator(mode="after")
    def validate_adjustments(self) -> Self:
        if self.adjustments:
            assert (
                not self.peft_config
            ), "PEFT should not be used outside model adjustments. Pass it as part of model adjustments."
        return self
