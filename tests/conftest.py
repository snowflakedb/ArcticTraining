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
from pathlib import Path

import pytest

# Track which dist environment (cpu or gpu) is currently setup so we can avoid
# re-initializing.
pytest.current_dist_env = ""


def pytest_configure(config):
    # Assume cpu dist by default, but this will be torn down if only running gpu tests
    _setup_cpu_dist()


# Helper function because we can't directly call a fixture in pytest_configure
def _setup_cpu_dist():
    print("\n[Fixture] Setting up for 'cpu' tests")
    os.environ["DS_ACCELERATOR"] = "cpu"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_SIZE"] = "1"

    from deepspeed.comm import init_distributed

    init_distributed(auto_mpi_discovery=False)
    pytest.current_dist_env = "cpu"


@pytest.fixture(scope="function")
def setup_cpu_dist():
    if pytest.current_dist_env and pytest.current_dist_env != "cpu":
        print("\n[Fixture] Tearing down for current env (in cpu)")
        import deepspeed.comm as dist

        dist.barrier()
        dist.destroy_process_group()
        print("\n[Fixture] Tear down complete")
    elif pytest.current_dist_env == "cpu":
        return
    _setup_cpu_dist()


def _setup_gpu_dist():
    print("\n[Fixture] Setting up for 'gpu' tests")
    os.environ["DS_ACCELERATOR"] = "cuda"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_SIZE"] = "1"

    from deepspeed.comm import init_distributed

    init_distributed(auto_mpi_discovery=False)
    pytest.current_dist_env = "gpu"


@pytest.fixture(scope="function")
def setup_gpu_dist():
    if pytest.current_dist_env and pytest.current_dist_env != "gpu":
        print("\n[Fixture] Tearing down for current env (in gpu)")
        import deepspeed.comm as dist

        dist.barrier()
        dist.destroy_process_group()
        print("\n[Fixture] Tear down complete")
    elif pytest.current_dist_env == "gpu":
        return
    _setup_gpu_dist()


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    # Group tests by 'cpu' and 'gpu' markers
    cpu_tests = [item for item in items if "cpu" in item.keywords]
    gpu_tests = [item for item in items if "gpu" in item.keywords]

    # Reorder tests: all 'cpu' tests first, then 'gpu' tests
    items[:] = cpu_tests + gpu_tests


@pytest.fixture(scope="session")
def helpers_code_path() -> str:
    from . import test_helpers  # noqa: F401

    return str(Path(__file__).parent / "test_helpers.py")
