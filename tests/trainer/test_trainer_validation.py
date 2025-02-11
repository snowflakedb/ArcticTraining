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

import pytest

from arctic_training.config.trainer import TrainerConfig


@pytest.mark.cpu
def test_unregistered_trainer(tmp_path):
    config_dict = {
        "type": "unregistered_or_nonexistent",
        "exit_iteration": 2,
        "micro_batch_size": 1,
        "model": {
            "type": "random-weight-hf",
            "name_or_path": "HuggingFaceTB/SmolLM-135M-Instruct",
            "attn_implementation": "eager",
            "dtype": "float32",
        },
        "data": {
            "max_length": 2048,
            "sources": ["HuggingFaceH4/ultrachat_200k-truncated"],
        },
        "deepspeed": {"zero_optimization": {"stage": 0}},
        "optimizer": {"type": "cpu-adam"},
    }

    # # Fails in previous implementation of `TrainerConfig.parse_sub_config`, despite
    # # the implementation intending for this to succeed.
    # with pytest.raises(ValueError) as ctx:
    #     _config = TrainerConfig(**config_dict)  # noqa: F841
    #     assert (
    #         ctx.value.args[0]
    #         == "unregistered_or_nonexisten is not a registered Trainer."
    #     )

    with pytest.raises(KeyError) as ctx:
        _config = TrainerConfig(**config_dict)  # noqa: F841
        assert (
            ctx.value.args[0]
            == "Unable to validate info.field_name='model' because trainer type"
            " 'unregistered_or_nonexistent' is not registered. Please register the"
            " trainer and try again."
        )
