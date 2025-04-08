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

import logging
from pathlib import Path
from typing import List

import fire
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pandas.util import hash_array
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def main(
    in_dirs: List[str | Path],
    sub_dirs: List[str],
    out_dir: str | Path,
) -> None:
    """CLI entrypoint for combining multiple datasets into a single dataset.

    NOTE: Ids will be deterministically scrambled to mitigate collisions across
    different source datasets.
    """
    # Validate inputs.
    out_dir = Path(out_dir)
    in_dirs_paths = [Path(p) for p in in_dirs]
    del in_dirs
    for dataset_dir in in_dirs_paths:
        assert dataset_dir.is_dir(), f"{dataset_dir} is not a directory"

    # Combine.
    for subdir in sub_dirs:
        table_chunks = []
        for dataset_i, in_dir in enumerate(tqdm(in_dirs_paths, desc=f"combining {subdir}", unit="dataset")):
            # Read the parquet file for this dataset.
            path = in_dir / subdir
            table_chunk = pq.read_table(path)

            # Scramble all id values for global uniqueness across sources.
            id_columns = [(i, c) for i, c in enumerate(table_chunk.column_names) if c.endswith("ID")]
            for idc_i, idc_name in id_columns:
                idc_values = table_chunk.column(idc_i)
                scrambled = pa.array(scramble_ids(idc_values.to_numpy(), offset_seed=dataset_i))
                table_chunk = table_chunk.set_column(idc_i, idc_name, scrambled)

            table_chunks.append(table_chunk)

        # Combine all chunks.
        table = pa.concat_tables(table_chunks)

        # Validate ids are unique where they should be.
        if "queries.parquet" in subdir or "documents.parquet" in subdir:
            print(f"Validating ids are unique in combined {subdir}")
            for idc_i, idc_name in id_columns:
                assert table.column(idc_i).to_pandas().is_unique, f"{subdir=} {idc_name=}"

        out_path = out_dir / subdir
        print(f"Writing {out_path}")
        out_path.parent.mkdir(exist_ok=True, parents=True)
        pq.write_table(table, out_path)
        del table


def scramble_ids(original_ids: NDArray[np.uint64], offset_seed: int) -> NDArray[np.uint64]:
    """Add a deterministic pseudorandom offset to a set of ids, then re-hash them.

    This process should convert multiple sequences of non-random ids (e.g. incrementing
    integers 1,2, 3...) into a more randomly-distributed range in a deterministic
    manner.

    By varying the offset seed, one can map the same set of original ids to different
    random spreads, avoiding collisions.
    """
    original_ids = original_ids.astype(np.uint64)
    rng = np.random.default_rng(offset_seed)
    offset = np.frombuffer(rng.bytes(np.uint64().nbytes), dtype=np.uint64)[0]
    offset_ids = np.add(original_ids, offset, casting="same_kind")
    return hash_array(offset_ids)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    with logging_redirect_tqdm():
        fire.Fire(main)
