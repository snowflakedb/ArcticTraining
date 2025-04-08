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

import shutil
import tempfile
import urllib.request
from multiprocessing.pool import Pool
from os.path import basename
from os.path import isdir
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.util import hash_array
from tqdm.auto import tqdm

WIKI_ZIP_URL = "https://fever.ai/download/fever/wiki-pages.zip"
TRAIN_JSONL_URL = "https://fever.ai/download/fever/train.jsonl"
NUM_PROC_FOR_READ_DOC = 8
DATA_DIR = Path(__file__).parents[1] / "data"


def download(url, local_dir):
    assert isdir(local_dir)
    local_path = join(local_dir, basename(url))
    with urllib.request.urlopen(url) as response, open(local_path, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    return local_path


def read_wiki_dump_file(jsonl_filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_json(jsonl_filepath, lines=True)
    df.drop_duplicates(subset=["id"], inplace=True)
    df = df.loc[df["id"] != ""]
    texts = df["text"].to_numpy()
    ids = hash_array(df["id"].to_numpy())
    return ids, texts


if __name__ == "__main__":
    out_dir = DATA_DIR / "fever" / "text"
    out_dir.mkdir(exist_ok=True, parents=True)

    # Load the wiki documents.
    ids_chunks, text_chunks = [], []
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir_name:
        print(f"Downloading {WIKI_ZIP_URL} to {tmp_dir_name}")
        local_zip_file = download(WIKI_ZIP_URL, tmp_dir_name)
        print(f"Unzipping {local_zip_file}")
        shutil.unpack_archive(local_zip_file, tmp_dir_name)
        jsonl_filepaths = sorted((Path(tmp_dir_name) / "wiki-pages").iterdir())
        print(f"Got {len(jsonl_filepaths)} wiki dump files")
        with Pool(NUM_PROC_FOR_READ_DOC) as pool:
            res_iter = pool.imap(read_wiki_dump_file, jsonl_filepaths)
            res_iter = tqdm(
                res_iter,
                total=len(jsonl_filepaths),
                unit="file",
                desc="reading documents",
            )
            for id_chunk, text_chunk in res_iter:
                ids_chunks.append(id_chunk)
                text_chunks.append(text_chunk)
    dids = np.concatenate(ids_chunks)
    dids, idx_unique = np.unique(dids, return_index=True)
    texts = np.concatenate(text_chunks)[idx_unique]
    table_document = pa.table(
        {
            "DOCUMENT_ID": dids,
            "DOCUMENT_TEXT": texts,
        }
    )

    # Load the question data.
    print(f"Reading `{TRAIN_JSONL_URL}`")
    df_facts = pd.read_json(TRAIN_JSONL_URL, lines=True)
    df_verifiable = df_facts.loc[df_facts["verifiable"] == "VERIFIABLE"]

    # Get queries from unique claims.
    print("Processing claims into queries")
    query_texts = df_verifiable["claim"].to_numpy()
    qids = hash_array(query_texts)

    # Get labels from evidence.
    # Iterate once to get all the unique titles.
    all_titles = set()
    for evidence_list in tqdm(
        df_verifiable["evidence"],
        desc="finding all unique evidence titles",
        unit="question",
    ):
        for sub_list in evidence_list:
            for _annotation_id, _evidence_id, article_title, _sentence_id in sub_list:
                all_titles.add(article_title)

    # Map titles to uint64 document ids by hashing.
    all_titles_array = np.asarray(list(all_titles))
    title_map = dict(zip(all_titles_array, hash_array(all_titles_array)))

    # Get all the qid-did annotations in a second iteration of the evidence column.
    relevance_qids = []
    relevance_dids: list[int] = []
    dids_set = set(dids)
    num_skipped = 0
    is_kept_query = np.ones(len(qids), dtype=np.bool_)
    for i, (qid, evidence_list) in enumerate(
        zip(
            tqdm(qids, desc="parsing labels", unit="question"),
            df_verifiable["evidence"],
            strict=True,
        )
    ):
        relevant_dids = set()
        for sub_list in evidence_list:
            for _annotation_id, _evidence_id, article_title, _sentence_id in sub_list:
                did = title_map[article_title]
                if did in dids_set:
                    relevant_dids.add(did)
                # NOTE: A few docs might be missing from the dump.
                else:
                    num_skipped += 1
        if len(relevant_dids) > 0:
            relevance_qids.extend([qid] * len(relevant_dids))
            relevance_dids.extend(relevant_dids)
        else:
            is_kept_query[i] = False
    table_labels = pa.table(
        {
            "QUERY_ID": np.asarray(relevance_qids, dtype=np.uint64),
            "DOCUMENT_ID": np.asarray(relevance_dids, dtype=np.uint64),
        }
    )

    # Filter out queries with no documents and dedupe to construct query table.
    qids = qids[is_kept_query]
    query_texts = query_texts[is_kept_query]
    qids, unique_idx = np.unique(qids, return_index=True)
    query_texts = query_texts[unique_idx]
    table_query = pa.table({"QUERY_ID": qids, "QUERY_TEXT": query_texts})
    print(
        f"Skipped {num_skipped:,} annotations due to missing documents, "
        f"triggering {(~is_kept_query).sum():,} skipped queries which had no documents."
    )

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
