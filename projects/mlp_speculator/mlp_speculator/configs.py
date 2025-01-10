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

import json
import os

import transformers


# Configs used for savings model checkpoint for inference
class MLPSpeculatorConfig:
    """
    This is a simple MLP-based speculator Config.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of the input vector from the base model.
    inner_dim : int
        Latent dimensionality of the speculator model.
    vocab_size : int
        Number of entries in the tokenizer associated with the base model.
    n_predict : int
        Number of heads / number of tokens to guess ahead. Model size and speed scale with this value.
    tie_weights : bool
        If true, use a single set of weights for every model head/stage after the first.
        The initial projection from the base model may have a different size, so that stays separate.
    scale_input: bool
        If true, apply an extra layernorm to the initial state vector input.
        Helps training dynamics, particularly when base model output has unusual scale.
    """

    def __init__(
        self,
        base_model_name_or_path,
        emb_dim,
        inner_dim,
        vocab_size,
        n_predict,
        tie_weights=False,
        scale_input=False,
    ):
        self.architectures = "MLPSpeculatorPreTrainedModel"
        self.base_model_name_or_path = base_model_name_or_path

        self.emb_dim = emb_dim
        self.inner_dim = inner_dim
        self.model_type = "mlp_speculator"

        self.n_candidates = n_predict
        self.n_predict = n_predict

        self.scale_input = scale_input
        self.tie_weights = tie_weights
        self.top_k_tokesns_per_head = [1 for i in range(self.n_predict)]

        self.torch_dtype = "bfloat16"
        self.transformers_version = transformers.__version__
        self.vocab_size = vocab_size

    def save(self, output_dir):
        save_path = os.path.join(output_dir, "config.json")
        with open(save_path, "w") as f:
            json.dump(self.__dict__, f)
