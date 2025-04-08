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

"""Since `sentence-transformers/hotpotqa` appears to have substantially different
documents than those in the official HotpotQA training data, we provide this
alternative import script to pull from their data.
"""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from pandas.util import hash_array
from tqdm.auto import tqdm

DATA_DIR = Path(__file__).parents[1] / "data"


if __name__ == "__main__":
    out_dir = DATA_DIR / "hotpotqa" / "text"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load the data.
    print("Loading data from `sentence-transformers/hotpotqa`")
    df_triplet = load_dataset("sentence-transformers/hotpotqa", "triplet-all", split="train").to_pandas()

    # Get the unique docs.
    print("Finding the unique docs")
    unique_chunks = [df_triplet["positive"].unique(), df_triplet["negative"].unique()]
    doc_texts_array = np.unique(np.concatenate(unique_chunks))
    print("Getting doc ids via hashing")
    doc_ids_array = hash_array(doc_texts_array)
    doc_text_to_id = dict(zip(doc_texts_array, doc_ids_array))

    # Get the unique queries.
    print("Finding the unique queries")
    query_texts_array = df_triplet["anchor"].unique()
    query_ids_array = hash_array(query_texts_array)
    query_text_to_id = dict(zip(query_texts_array, query_ids_array))

    # Get the unique relations.
    unique_pairs: set[tuple[int, int]] = set()
    for query, positive in tqdm(
        df_triplet[["anchor", "positive"]].itertuples(index=False),
        desc="Finding unique pairs",
        total=len(df_triplet),
    ):
        query_id = query_text_to_id[query]
        positive_id = doc_text_to_id[positive]
        unique_pairs.add((query_id, positive_id))
    relation_qids, relation_dids = zip(*unique_pairs)
    relation_qids_array = np.asarray(relation_qids, dtype=np.uint64)
    relation_dids_array = np.asarray(relation_dids, dtype=np.uint64)
    del unique_pairs

    # Put together the tables.
    table_query = pa.table({"QUERY_ID": query_ids_array, "QUERY_TEXT": query_texts_array})
    table_document = pa.table({"DOCUMENT_ID": doc_ids_array, "DOCUMENT_TEXT": doc_texts_array})
    table_labels = pa.table({"QUERY_ID": relation_qids_array, "DOCUMENT_ID": relation_dids})

    # Ensure all relation ids are in the query and doc tables.
    missing_doc_ids = np.setdiff1d(
        table_labels["DOCUMENT_ID"].to_numpy(),
        table_document["DOCUMENT_ID"].to_numpy(),
    )
    missing_query_ids = np.setdiff1d(table_labels["QUERY_ID"].to_numpy(), table_query["QUERY_ID"].to_numpy())
    assert missing_doc_ids.shape[0] == 0
    assert missing_query_ids.shape[0] == 0

    # Write tables to disk.
    print(f"Writing data to `{out_dir}`")
    pq.write_table(table_query, out_dir / "queries.parquet")
    pq.write_table(table_document, out_dir / "documents.parquet")
    pq.write_table(table_labels, out_dir / "labels.parquet")
    print("All done!")
