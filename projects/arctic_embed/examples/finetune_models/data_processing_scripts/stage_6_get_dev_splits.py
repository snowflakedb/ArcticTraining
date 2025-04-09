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
import shutil
from pathlib import Path

NUM_EVAL_BATCHES = 5
DATA_ROOT = Path("/scratch/ArcticTraining/projects/arctic_embed/examples/finetune_models/data/")
PRETOKENIZED_DATASAET = "example_dot95"

if __name__ == "__main__":
    sub_dirs = sorted(x for x in DATA_ROOT.iterdir() if x.is_dir() and x.name not in ("combined", "eval"))
    data_dirs = [sd / "pretokenized" / PRETOKENIZED_DATASAET / "data" for sd in sub_dirs]
    out_dirs = [DATA_ROOT / "eval" / sd.name for sd in sub_dirs]

    # Validate that all datasets have the same tokenization metadata.
    consensus_metadata = None
    for dd in data_dirs:
        metadata_path = dd.parent / "metadata.json"
        assert metadata_path.exists(), f"{metadata_path} does not exist"
        metadata = json.loads(metadata_path.read_text())
        if consensus_metadata is None:
            consensus_metadata = metadata
        else:
            if consensus_metadata != metadata:
                msg = f"{metadata_path} does not match consensus {consensus_metadata=} {metadata=}"
                raise ValueError(msg)

    # Copy the metadata.
    out_metadata_path = out_dirs[0].parent / "metadata.json"
    out_metadata_path.parent.mkdir(exist_ok=True, parents=True)
    out_metadata_path.write_text(json.dumps(consensus_metadata, indent=2))

    # Copy a few batches from each dataset to use as eval.
    for in_dir, out_dir in zip(data_dirs, out_dirs, strict=True):
        for batch_dir in sorted(in_dir.iterdir())[:NUM_EVAL_BATCHES]:
            shutil.copytree(batch_dir, out_dir / batch_dir.name)
