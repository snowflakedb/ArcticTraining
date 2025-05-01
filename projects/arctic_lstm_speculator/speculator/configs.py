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
class ArcticLSTMSpeculatorConfig:
    """
    This is a simple MLP-based speculator Config.
    """

    def __init__(
        self,
        base_model_name_or_path,
        input_hidden_dim,
        inner_dim,
        proj_dim,
        emb_dim,
        vocab_size,
        n_predict,
        tie_weights=False,
        scale_input=False,
        method="sum_rnn",
        tie_lstm_embs=False,
    ):
        self.architectures = ["ArcticLSTMSpeculatorPreTrainedModel"]
        self.base_model_name_or_path = base_model_name_or_path

        self.input_hidden_dim = input_hidden_dim
        self.inner_dim = str(inner_dim)
        self.proj_dim = str(proj_dim)
        self.emb_dim = str(emb_dim)
        self.model_type = "mlp_speculator"

        self.n_candidates = n_predict
        self.n_predict = n_predict

        self.scale_input = scale_input
        self.tie_weights = tie_weights
        self.tie_lstm_embs = tie_lstm_embs
        self.top_k_tokens_per_head = [1 for i in range(self.n_predict)]

        self.torch_dtype = "bfloat16"
        self.transformers_version = transformers.__version__
        self.vocab_size = vocab_size
        self.method = method

    def save(self, output_dir):
        save_path = os.path.join(output_dir, "config.json")
        with open(save_path, "w") as f:
            json.dump(self.__dict__, f)
