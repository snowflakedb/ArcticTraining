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
from typing import Optional
from typing import Type
from typing import Union

from arctic_training.config.base import BaseConfig
from arctic_training.registry import get_registered_tokenizer_factory

if TYPE_CHECKING:
    from arctic_training.tokenizer.factory import TokenizerFactory


class TokenizerConfig(BaseConfig):
    type: str = ""
    """ Tokenizer factory type. Defaults to the `tokenizer_factory_type` of the trainer. """

    name_or_path: Optional[Union[str, Path]] = ""
    """ Tokenizer name (as described in Hugging Face model hub) or local path directory containing tokenizer. """

    @property
    def factory(self) -> Type["TokenizerFactory"]:
        return get_registered_tokenizer_factory(name=self.type)
