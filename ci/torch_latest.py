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

import modal

ROOT_PATH = Path(__file__).parents[1]

# flash_attn_release = (
#     "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3"
#     "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp313-cp313-linux_x86_64.whl"
# )

# fmt: off
image = (
    modal.Image
    .from_registry("pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel", add_python="3.12")
    # XXX: add a freeze requirements to get the caching working
    .add_local_dir(ROOT_PATH, remote_path="/root/", copy=True)
    # ci-requirements.txt is generated in the github workflow job which allows us to skip image rebuilding if the requirements haven't changed since the last CI job was run
    .uv_pip_install_from_requirements(ROOT_PATH / "ci-requirements.txt", gpu="any")
    .uv_pip_install("flash_attn", gpu="any", extra_options="--no-build-isolation")
    # .uv_pip_install_from_requirements(ROOT_PATH / "ci-requirements2.txt", gpu="any", extra_options="--no-build-isolation")
    .run_commands("uv pip install --system /root")
)
# fmt: on

app = modal.App("arctictraining-torch-latest-ci", image=image)


@app.function(
    gpu="l40s:2",
    timeout=300,
    # timeout=1800,
)
def pytest():
    import subprocess

    # XXX: need to re-add `-n 4` when hardwired deepspeed dist init is removed from conftest.py - it conflicts with concurrent test runs as it assigns the same port to all tests
    cmd = "pytest --disable-warnings --instafail -m gpu --verbose tests/test_ulysses_alst.py"
    # cmd = "pytest --disable-warnings --instafail -m gpu --verbose tests"

    print(f"Running: {cmd}")
    subprocess.run(
        cmd.split(),
        check=True,
        cwd=ROOT_PATH / ".",
    )
