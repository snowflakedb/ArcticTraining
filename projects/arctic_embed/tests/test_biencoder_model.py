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

import torch
from arctic_embed.core.biencoder_model import average_pool
from arctic_embed.core.biencoder_model import first_token_pool
from arctic_embed.core.biencoder_model import last_token_pool
from torch.testing import assert_close

attention_left_pad = torch.tensor(
    [
        [False, False, True, True],
        [True, True, True, True],
    ]
)
values_left_pad = torch.tensor(
    [
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        [
            [11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0],
            [14.0, 14.0, 14.0],
        ],
    ]
)
attention_right_pad = attention_left_pad.flip(dims=[1])
values_right_pad = values_left_pad.flip(dims=[1])


def test_pooling() -> None:
    want_first_or_last = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [11.0, 11.0, 11.0],
        ]
    )
    want_average = torch.tensor(
        [
            [1.5, 1.5, 1.5],
            [12.5, 12.5, 12.5],
        ]
    )
    first_of_left_pad = first_token_pool(values_left_pad, attention_left_pad)
    assert_close(first_of_left_pad, want_first_or_last)
    last_of_right_pad = last_token_pool(values_right_pad, attention_right_pad)
    assert_close(last_of_right_pad, want_first_or_last)
    assert_close(
        average_pool(values_left_pad, attention_left_pad),
        average_pool(values_right_pad, attention_right_pad),
    )
    assert_close(average_pool(values_left_pad, attention_left_pad), want_average)
