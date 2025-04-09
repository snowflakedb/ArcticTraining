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

"""Abstractions for models used in the bulk embedding step of hard negative mining."""

import logging
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Sequence

import torch
from torch import Tensor
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .utils import NDArrayOfFloat
from .utils import first_token_pool

logger = logging.getLogger(__name__)


def log(msg: str) -> None:
    logger.info(msg)


class AbstractEmbedder(metaclass=ABCMeta):
    @abstractmethod
    def to_device(self, device: torch.device | str) -> None:
        pass

    @abstractmethod
    def embed_batch(self, texts: Sequence[str], is_query: bool, **kwargs: Any) -> NDArrayOfFloat:
        pass


class Arctic2LargeEmbedder(AbstractEmbedder):
    HF_MODEL_ID: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    QUERY_PREFIX: str = "query: "
    DOC_PREFIX: str = ""

    def __init__(self, max_seq_len: int = 8192):
        log(f"Loading tokenizer and model model from HuggingFace: `{self.HF_MODEL_ID}`")
        self.model = AutoModel.from_pretrained(self.HF_MODEL_ID)
        self.device: str | torch.device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def to_device(self, device: str | torch.device) -> None:
        self.device = device
        self.model = self.model.to(device)

    def tokenize(self, texts: Sequence[str], prefix: str, max_seq_len: int | None) -> dict[str, Tensor]:
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        inputs: dict[str, Tensor] = self.tokenizer(
            prefixed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        return inputs

    def embed_batch(
        self,
        texts: Sequence[str],
        is_query: bool,
        **_kwargs: Any,  # For type alignment with parent class.
    ) -> NDArrayOfFloat:
        inputs = self.tokenize(
            texts=texts,
            prefix=self.QUERY_PREFIX if is_query else self.DOC_PREFIX,
            max_seq_len=self.max_seq_len,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        seq_of_vec = outputs.last_hidden_state
        vec = first_token_pool(seq_of_vec, inputs["attention_mask"])
        return vec.cpu().numpy()
