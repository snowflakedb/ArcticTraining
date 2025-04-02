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

import html
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.util import hash_array
from tqdm.auto import tqdm

RAW_DATA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
DATA_DIR = Path(__file__).parents[1] / "data"


def normalize_string(s: str) -> str:
    """Normalize a string by unescaping HTML entities and normalizing unicode."""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKD", s)
    return s


if __name__ == "__main__":
    out_dir = DATA_DIR / "hotpotqa" / "text"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Download data and load from disk to memory.
    print(f"Loading data from {RAW_DATA_URL}")
    df_hotpot = pd.read_json(RAW_DATA_URL)

    # Pull out all the unique context documents.
    doc_titles = []
    docs = []
    for item in tqdm(df_hotpot["context"], desc="Parsing documents"):
        for title, body_list in item:
            body = " ".join(body_list)
            # NOTE: Without normalizing strings, we can get slightly different doc texts for the same title.
            title = normalize_string(title)
            body = normalize_string(body)
            doc = title + "\n" + body
            docs.append(doc)
            doc_titles.append(title)
    df_document = pd.DataFrame({"DOCUMENT_ID": hash_array(np.asarray(doc_titles)), "DOCUMENT_TEXT": docs})
    df_document.drop_duplicates(subset=["DOCUMENT_ID"], inplace=True)
    assert df_document["DOCUMENT_ID"].is_unique
    assert df_document["DOCUMENT_TEXT"].is_unique
    table_document = pa.Table.from_pandas(df_document, preserve_index=False)
    del df_document

    # Pull out all the questions, deduping.
    print("Parsing questions as queries")
    questions = df_hotpot["question"].apply(normalize_string).to_numpy()
    qids = hash_array(questions)
    qids, unique_idx = np.unique(qids, return_index=True)
    questions = questions[unique_idx]
    table_query = pa.table({"QUERY_ID": qids, "QUERY_TEXT": questions})

    # Pull out the relation labels.
    relation_qids = []
    relation_doc_titles = []
    for qid, facts in zip(
        tqdm(qids, desc="Parsing labels"),
        df_hotpot.loc[:, "supporting_facts"].iloc[unique_idx],
        strict=True,
    ):
        if len(facts) == 0:
            print("skipping")
            continue
        fact_doc_titles = np.asarray(list(set(normalize_string(title) for title, _sentence_idx in facts)))
        for doc_title in fact_doc_titles:
            relation_qids.append(qid)
            relation_doc_titles.append(doc_title)
    relation_qids_array = np.asarray(relation_qids, dtype=np.uint64)
    del relation_qids
    relation_dids = hash_array(np.asarray(relation_doc_titles))
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
    print(f"Reducing labels from {len(qids):,} to {len(table_labels):,} by deduping.")
    print(f"Writing data to `{out_dir}`")
    pq.write_table(table_query, out_dir / "queries.parquet")
    pq.write_table(table_document, out_dir / "documents.parquet")
    pq.write_table(table_labels, out_dir / "labels.parquet")
    print("All done!")
