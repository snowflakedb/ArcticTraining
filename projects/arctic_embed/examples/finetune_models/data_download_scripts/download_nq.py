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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from pandas.util import hash_array

DATA_DIR = Path(__file__).parents[1] / "data"


if __name__ == "__main__":
    out_dir = DATA_DIR / "nq" / "text"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Download data and load from disk to memory.
    print("Reading `sentence-transformers/natural-questions` from HuggingFace")
    nq_train = load_dataset("sentence-transformers/natural-questions", split="train")

    # Create qids and docids as hashes of the texts.
    qids = hash_array(np.asarray(nq_train["query"]))
    dids = hash_array(np.asarray(nq_train["answer"]))

    # Dedupe queries and docs.
    qids_unique, qidx = np.unique(qids, return_index=True)
    dids_unique, didx = np.unique(dids, return_index=True)
    print(f"Reducing queries from {len(qids):,} to {len(qids_unique):,} by deduping.")
    print(f"Reducing documents from {len(dids):,} to {len(dids_unique):,} by deduping.")

    # Construct tables and write to disk.
    table_query = pa.table({"QUERY_ID": qids_unique, "QUERY_TEXT": np.asarray(nq_train["query"])[qidx]})
    table_document = pa.table(
        {
            "DOCUMENT_ID": dids_unique,
            "DOCUMENT_TEXT": np.asarray(nq_train["answer"])[didx],
        }
    )
    table_labels = pa.Table.from_pandas(
        pd.DataFrame({"QUERY_ID": qids, "DOCUMENT_ID": dids}).drop_duplicates(keep="first"),
        preserve_index=False,
    )
    print(f"Reducing labels from {len(qids):,} to {len(table_labels):,} by deduping.")
    print(f"Writing data to `{out_dir}`")
    pq.write_table(table_query, out_dir / "queries.parquet")
    pq.write_table(table_document, out_dir / "documents.parquet")
    pq.write_table(table_labels, out_dir / "labels.parquet")
    print("All done!")
