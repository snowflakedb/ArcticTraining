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

from arctic_training.config.base import BaseConfig
from arctic_training.config.utils import HumanFloat
from arctic_training.config.utils import HumanInt
from arctic_training.config.utils import parse_human_val


@pytest.mark.parametrize(
    "value, expected",
    [
        # Regular int
        ("1_000_000", 1_000_000),
        # Exponential
        ("10^6", 1_000_000),
        ("1e6", 1_000_000),
        ("1.1e6", 1_100_000),
        ("10e-1", 1),
        # Suffixes
        ("1K", 1_000),
        ("1.1K", 1_100),
        ("1Ki", 2**10),
    ],
)
def test_human_int(value, expected):
    class TestConfig(BaseConfig):
        val: HumanInt

    assert TestConfig(val=value).val == expected, f"Failed for input '{value}', expected parsed value: {expected}"

    neg_value = f"-{value}"
    assert (
        TestConfig(val=neg_value).val == -expected
    ), f"Failed for input '{neg_value}', expected parsed value: {-expected}"


@pytest.mark.parametrize(
    "value, expected",
    [
        # Regular float
        ("1.5", 1.5),
        # Percentage
        ("1.5%", 0.015),
        # Exponential
        ("10.1^3", 10.1**3),
        ("1e-3", 0.001),
        ("1.1e-3", 0.0011),
        # Suffixes
        ("1K", 1_000.0),
        ("1.1K", 1_100.0),
        ("1Ki", 2**10),
    ],
)
def test_human_float(value, expected):
    class TestConfig(BaseConfig):
        val: HumanFloat

    assert TestConfig(val=value).val == expected, f"Failed for input '{value}', expected parsed value: {expected}"

    neg_value = f"-{value}"
    assert (
        TestConfig(val=neg_value).val == -expected
    ), f"Failed for input '{neg_value}', expected parsed value: {-expected}"


@pytest.mark.parametrize("value", ["1.5", "10%", "1e-3", "1.00001K"])
def test_human_int_invalid(value):
    class TestConfig(BaseConfig):
        val: HumanInt

    with pytest.raises(Exception):
        TestConfig(val=value)


@pytest.mark.parametrize("value", ["err", "1Ke3", "1^10e2", "1K^2", "1K^2e3"])
def test_human_val_invalid_syntax(value):
    with pytest.raises(Exception):
        parse_human_val(value)
