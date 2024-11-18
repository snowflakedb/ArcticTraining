from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from arctic_training.logging import logger
from arctic_training.register import get_dataset_class

from .base import BaseConfig


class DataConfig(BaseConfig):
    datasets: List[str]
    eval_datasets: List[str] = []
    train_eval_split: Tuple[float, float] = (1.0, 0.0)
    tokenizer_name_or_path: Optional[str] = Field(None, alias="tokenizer")
    max_length: int = 8192
    eval_frequency: int = 0
    num_proc: int = 16
    mask_inputs: bool = True
    seed: int = 42
    bypass_tokenizer_verification: bool = False
    not_packing_input: bool = False
    use_data_cache: bool = True
    cache_processed_data: bool = False
    data_cache_dir: Path = Path("/tmp/")
    dataset_type: str = "sft"

    @model_validator(mode="after")
    def validate_cache_dir(self) -> Self:
        if self.use_data_cache:
            assert (
                self.data_cache_dir is not None
            ), "You must provide a data_cache_dir if use_data_cache is True."
            if not self.data_cache_dir.exists():
                logger.warning(
                    f"Caching directory {self.data_cache_dir} does not exist. Creating it."
                )
                self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def validate_dataset_type_settings(self) -> Self:
        if self.dataset_type == "dpo":
            assert self.mask_inputs, "DPO dataset should have masked input."
        if not self.not_packing_input:
            assert (
                self.dataset_type == "sft"
            ), "Packing input is only supported for SFT dataset."
        return self

    @model_validator(mode="after")
    def validate_train_eval_split(self) -> Self:
        if self.eval_datasets:
            assert (
                self.train_eval_split[0] == 1.0
            ), "train_eval_split should be (1.0, 0.0) when eval_datasets is provided."
        if self.train_eval_split[1] > 0.0:
            assert (
                not self.eval_datasets
            ), "If you provide the evaluation split, you should not provide the evaluation datasets."
        assert sum(self.train_eval_split) == 1.0, "train_eval_split should sum to 1.0."
        if self.eval_datasets or self.train_eval_split[1] > 0.0:
            assert (
                self.eval_frequency > 0
            ), "eval_frequency should be greater than 0 when using evaluation datasets."
        return self

    @model_validator(mode="after")
    def validate_data_loaders(self) -> Self:
        for dataset in self.datasets + self.eval_datasets:
            _ = get_dataset_class(dataset_type=self.dataset_type, dataset_name=dataset)
        return self

    @model_validator(mode="after")
    def validate_tokenizer(self) -> Self:
        return self
        # TODO: fix this
        supported_tokenizers = ["meta-llama-3.1"]
        if self.tokenizer_name_or_path is None:
            return self
        if self.bypass_tokenizer_verification:
            return self
        if not self.not_mask_input:
            assert (
                self.tokenizer_name_or_path in supported_tokenizers
            ), f"Tokenizer {self.tokenizer_name_or_path} is not supported. Please bypass the tokenizer verification if you are sure it is correct."
        return self
