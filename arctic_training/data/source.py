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

from abc import ABC
from abc import abstractmethod

from datasets import Dataset

from arctic_training.callback.mixin import CallbackMixin


class DataSource(ABC, CallbackMixin):
    name: str
    data_factory_type: str

    def __init__(self, num_proc: int, eval: bool) -> None:
        self.num_proc = num_proc
        self.eval = eval

    def __call__(self) -> Dataset:
        return self.load_fn(self.num_proc, self.eval)

    @abstractmethod
    def load_fn(self, num_proc: int, eval: bool) -> Dataset:
        raise NotImplementedError("load_fn must be implemented in subclass")
