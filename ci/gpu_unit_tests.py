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
    # later when things stability switch to a newer image and update torch/cuda versions in the yaml file
    # .from_registry("pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel", add_python="3.12")
    .from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel", add_python="3.12")
    # XXX: add a freeze requirements to get the caching working
    # ci-requirements.txt is generated in the github workflow job which allows us to skip image
    # rebuilding if the requirements haven't changed since the last CI job was run
    .pip_install_from_requirements(ROOT_PATH / "requirements-general.txt", gpu="any")
    .pip_install_from_requirements(ROOT_PATH / "requirements-torch.txt", gpu="any")
    .pip_install_from_requirements(ROOT_PATH / "requirements-flash_attn.txt", gpu="any", extra_options="--no-build-isolation")
    # add_local_dir copies the repo over - since we use copy=True (because we need always the
    # latest files), it has to be run as late as possible to allow caching of the previous stages -
    # so run it before installing the main repo itself.
    .add_local_dir(ROOT_PATH, remote_path="/root/", copy=True)
    .run_commands("uv pip install --system /root")
)
# fmt: on

app = modal.App("arctictraining-torch-latest-ci", image=image)


@app.function(
    gpu="l40s:2",
    timeout=1800,  # 1800sec=30min
)
def pytest():
    import os
    import subprocess

    # some debug helpers if needed to diagnose things
    # print(f"{os.environ.get('HF_TOKEN', 'NONE')=}")
    # subprocess.run(["find", str(ROOT_PATH)])
    # subprocess.run(["nvidia-smi"])
    # this overcomes mkl multi-process breakage (that is when using xdist)
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    # overcome CI log buffering to see tests reported in real time
    os.environ["PYTHONUNBUFFERED"] = "1"

    # another way to install the repo if not done via modal.Image - same efficiency since it'll have to be reinstalled every time CI runs anyway (dependencies will not be reinstalled as they have been cached in modal.Image)
    # cmd = "uv pip install --system ."
    # print(f"Running: {cmd}")
    # subprocess.run(
    #     cmd.split(),
    #     check=True,
    #     cwd=ROOT_PATH / ".",
    # )

    # XXX: need to re-add `-n 4` when hardwired deepspeed dist init is removed from conftest.py - it conflicts with concurrent test runs as it assigns the same port to all tests
    cmd = "pytest --disable-warnings --instafail -m gpu --verbose tests"
    # cmd = "pytest --disable-warnings --instafail -m gpu --verbose tests"

    print(f"Running: {cmd}")
    subprocess.run(
        cmd.split(),
        check=True,
        cwd=ROOT_PATH / ".",
    )
