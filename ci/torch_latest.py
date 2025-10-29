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

# fmt: off
image = (
    modal.Image
    .from_registry("pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel", add_python="3.12")
    # XXX: add a freeze requirements to get the caching working
    .add_local_dir(ROOT_PATH, remote_path="/root/", copy=True)
    .run_commands("uv pip install --system /root[testing]")
)
# fmt: on

app = modal.App("arctictraining-torch-latest-ci", image=image)


@app.function(
    gpu="l40s:2",
    timeout=1800,
)
def pytest():
    import subprocess

    subprocess.run(
        # XXX: need to re-add `-n 4` when hardwired deepspeed dist init is removed from conftest.py - it conflicts with concurrent test runs as it assigns the same port to all tests
        "pytest -n 4 --disable-warnings --instafail -m gpu --verbose tests".split(),
        check=True,
        cwd=ROOT_PATH / ".",
    )
