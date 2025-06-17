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

import os
import sys
from pathlib import Path

import pytest

# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = str(Path(__file__).resolve().parents[1])
sys.path.insert(1, git_repo_path)


def pytest_configure(config):
    # allow having multiple repository checkouts and not needing to remember to rerun
    # 'pip install .' when switching between checkouts and running tests.
    git_repo_path = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, git_repo_path)

    # TODO: Make it so that cpu and gpu tests can be run with a single command.
    # This requires some work with tearing down/setting up dist environments
    # that have not been worked out yet.
    if "not gpu" in config.option.markexpr:
        _setup_cpu_dist()
    else:
        _setup_gpu_dist()


def _setup_cpu_dist():
    os.environ["DS_ACCELERATOR"] = "cpu"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_SIZE"] = "1"

    from deepspeed.comm import init_distributed

    init_distributed(auto_mpi_discovery=False)


def _setup_gpu_dist():
    os.environ["DS_ACCELERATOR"] = "cuda"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_SIZE"] = "1"

    from deepspeed.comm import init_distributed

    init_distributed(auto_mpi_discovery=False)


# Eventually when we can run cpu + gpu tests, we will want to order the tests to
# avoid thrashing back and forth between the distirbuted environments.
"""
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    t scpu_tests = [item for item in items if "cpu" in item.keywords]
    gpu_tests = [item for item in items if "gpu" in item.keywords]

    # Reorder tests: all 'cpu' tests first, then 'gpu' tests
    items[:] = cpu_tests + gpu_tests
"""


# Load helper functions automatically for all tests
@pytest.fixture(scope="session", autouse=True)
def helpers_code_path() -> None:
    from . import helpers  # noqa: F401


@pytest.fixture(scope="session")
def model_name() -> str:
    return "hf-internal-testing/tiny-random-Olmo2ForCausalLM"
