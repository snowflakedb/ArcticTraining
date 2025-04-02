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
from tqdm.auto import tqdm

DATA_DIR = Path(__file__).parents[1] / "data"


def get_docs_table() -> pa.Table:
    # Download corpus data and load from disk to memory.
    print("Reading `Tevatron/msmarco-passage-corpus` from HuggingFace")
    corpus_train = load_dataset("Tevatron/msmarco-passage-corpus", split="train")

    # Combine title and body of doc texts.
    print("Re-structuring documents")
    doc_texts = np.full(len(corpus_train), None, dtype=object)
    for i, (title, passage) in enumerate(
        zip(
            corpus_train["title"],
            tqdm(corpus_train["text"], desc="processing documents", unit="doc"),
        )
    ):
        text = f"{title}\n{passage}" if title.strip() not in ("", "-") else passage
        doc_texts[i] = text

    # Convert doc ids to uint64 for consistency with hashed doc ids.
    print("Retyping ids to uint64")
    dids = np.asarray(corpus_train["docid"], dtype=np.uint64)

    # Write restructured docs to disk.
    print("Constructing document table")
    table_document = pa.table({"DOCUMENT_ID": dids, "DOCUMENT_TEXT": doc_texts})

    return table_document


def get_query_and_qrel_tables() -> tuple[pa.Table, pa.Table]:
    # Download query and relevance data and load from disk to memory.
    print("Reading `Tevatron/msmarco-passage` from HuggingFace")
    q_with_rel_train = load_dataset("Tevatron/msmarco-passage", split="train")
    qrel_qids: list[int] = []
    qrel_dids: list[int] = []
    for qid, rel_docs in zip(
        q_with_rel_train["query_id"],
        tqdm(q_with_rel_train["positive_passages"], desc="processing qrels", unit="query"),
    ):
        qrel_qids.extend([qid] * len(rel_docs))
        qrel_dids.extend([item["docid"] for item in rel_docs])

    # Convert ids to uint64 for consistency with hashed ids.
    print("Retyping ids to uint64")
    qids = np.asarray(q_with_rel_train["query_id"], dtype=np.uint64)
    qrel_qids = np.asarray(qrel_qids, dtype=np.uint64)  # type: ignore
    qrel_dids = np.asarray(qrel_dids, dtype=np.uint64)  # type: ignore

    # Construct tables.
    table_query = pa.table({"QUERY_ID": qids, "QUERY_TEXT": q_with_rel_train["query"]})
    table_labels = pa.table({"QUERY_ID": qrel_qids, "DOCUMENT_ID": qrel_dids})

    return table_query, table_labels


if __name__ == "__main__":
    out_dir = DATA_DIR / "msmarco" / "text"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Process document data.
    table_document = get_docs_table()

    # Process query and relevance data.
    table_query, table_labels = get_query_and_qrel_tables()

    # Dedupe doc texts.
    doc_texts_unique, idx, idx_inverse = np.unique(
        table_document["DOCUMENT_TEXT"], return_index=True, return_inverse=True
    )
    dids = table_document["DOCUMENT_ID"].to_numpy()
    dids_unique = dids[idx]
    print(f"Reducing documents from {len(table_document):,} to {len(doc_texts_unique):,} by deduping.")
    table_document = pa.table({"DOCUMENT_ID": dids_unique, "DOCUMENT_TEXT": doc_texts_unique})

    # Ensure queries are deduped.
    assert len(pd.unique(table_query["QUERY_ID"].to_numpy())) == len(table_query)
    assert len(pd.unique(table_query["QUERY_TEXT"].to_numpy())) == len(table_query)

    # Map label doc ids to the canonical deduped doc id.
    print("Mapping label doc ids to canonical deduped doc ids")
    dedup_did_map = dict(zip(dids, dids_unique[idx_inverse]))
    label_doc_ids = np.vectorize(dedup_did_map.get)(table_labels["DOCUMENT_ID"].to_numpy())

    # Dedupe query-doc pair labels.
    table_labels = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "QUERY_ID": table_labels["QUERY_ID"].to_numpy(),
                "DOCUMENT_ID": label_doc_ids,
            }
        ).drop_duplicates(keep="first"),
        preserve_index=False,
    )
    print(f"Reduced label pairs from {len(label_doc_ids):,} to {len(table_labels):,} by deduping.")

    # Write to disk.
    print(f"Writing data to `{out_dir}`")
    pq.write_table(table_query, out_dir / "queries.parquet")
    pq.write_table(table_document, out_dir / "documents.parquet")
    pq.write_table(table_labels, out_dir / "labels.parquet")
    print("All done!")
