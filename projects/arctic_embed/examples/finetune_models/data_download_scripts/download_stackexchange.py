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

from bisect import bisect
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm.auto import tqdm

RAW_DATA_URL = "https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/stackexchange_duplicate_questions_title-body_title-body.jsonl.gz"  # noqa: E501
DATA_DIR = Path(__file__).parents[1] / "data"


if __name__ == "__main__":
    out_dir = DATA_DIR / "stackexchange" / "text"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Download data and load from disk to memory.
    print(
        "Reading stackexchange title-body pairs from `sentence-transformers/embedding-training-data` from HuggingFace"
    )
    df = pd.read_json(
        RAW_DATA_URL,
        lines=True,
    )

    # Download a dataset variant with just the body texts so we can figure out
    # how to split title and body.
    print("Reading `sentence-transformers/stackexchange-duplicates` from Huggingface")
    df_bodies = load_dataset(
        "sentence-transformers/stackexchange-duplicates",
        name="body-body-pair",
        split="train",
    ).to_pandas()
    bodies = df_bodies["body1"].tolist() + df_bodies["body2"].tolist()
    sorted_rev_bodies = sorted(x[::-1] for x in bodies)

    def split_title_body(title_body_combo: str) -> tuple[str, str] | None:
        """Splits a title and body text by looking up the body for a concatenation
        of title and body.

        NOTE: We use binary search on a sorted list of reversed items to go much
        faster than brute force.
        """
        i = bisect(sorted_rev_bodies, title_body_combo[::-1]) - 1
        body = sorted_rev_bodies[i][::-1]
        if not title_body_combo.endswith(body):
            return None
        title = title_body_combo.removesuffix(body).rstrip()
        return title, body

    # Go through the dataset and split the title and body for each of the title-body
    # pairs of the duplicate entries.
    num_missing = 0
    ids = []
    query_texts = []
    doc_texts = []
    qrel_qids = []
    qrel_dids = []
    for i, (tb1, tb2) in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc="Splitting title-body")):
        res1 = split_title_body(tb1)
        res2 = split_title_body(tb2)
        if res1 is None or res2 is None:
            num_missing += 1
            continue

        # Unpack items and give ids.
        q1, d1 = res1
        q2, d2 = res2
        id1 = 2 * i
        id2 = id1 + 1

        # Add query and doc table values.
        ids.extend([id1, id2])
        query_texts.extend([q1, q2])
        doc_texts.extend([d1, d2])

        # Add relation table values.
        qrel_qids.extend([id1, id2])
        qrel_dids.extend([id2, id1])
    print(f"Number of rows lost due to failed body lookup: {num_missing:,}")
    table_query = pa.table({"QUERY_ID": np.asarray(ids, dtype=np.uint64), "QUERY_TEXT": query_texts})
    table_document = pa.table({"DOCUMENT_ID": np.asarray(ids, dtype=np.uint64), "DOCUMENT_TEXT": doc_texts})
    table_labels = pa.table(
        {
            "QUERY_ID": np.asarray(qrel_qids, dtype=np.uint64),
            "DOCUMENT_ID": np.asarray(qrel_dids, dtype=np.uint64),
        }
    )

    # Construct tables and write to disk.
    print(f"Writing data to `{out_dir}`")
    pq.write_table(table_query, out_dir / "queries.parquet")
    pq.write_table(table_document, out_dir / "documents.parquet")
    pq.write_table(table_labels, out_dir / "labels.parquet")
    print("All done!")
