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

"""Utilities for pre-tokenizing and batching data for training Arctic Embed models."""

from __future__ import annotations

import json
import logging
from os.path import join
from typing import Iterator
from typing import List
from typing import Optional
from typing import Type

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pydantic import BaseModel
from pydantic import ConfigDict
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from .many_parquet_dataset import ManyParquetDataset
from .many_parquet_dataset import get_fs_for_path

logger = logging.getLogger(__name__)


class ShardedJobInfo(BaseModel):
    # Pydantic config.
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_default=True,
        use_attribute_docstrings=True,
        populate_by_name=True,
    )

    # Other fields.
    shard_id: int
    num_shards: int


class PretokenizeAndBatchPairsConfig(BaseModel):
    # Pydantic config.
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_default=True,
        use_attribute_docstrings=True,
        populate_by_name=True,
    )

    # Other fields.
    tokenizer_id: str
    """Unique identifier of a tokenizer on the huggingface hub."""

    query_doc_pair_dir: str
    """Directory containing parquet files with query-document pairs to process."""

    output_dir: str
    """Directory to write the output to."""

    query_prefix: str
    """Prefix to add to queries before tokenizing. Can be the empty string."""

    document_prefix: str
    """Prefix to add to documents before tokenizing. Can be the empty string."""

    batch_size: int = 32_000
    """Number of query-document pairs to process in each batch. If there are duplicates
    in the input data, the actual size of output batches may be smaller."""

    max_seq_length: int = 1024
    """The maximum sequence length to truncate to during pretokenization."""

    pre_truncate_max_chars_per_token: int = 8
    """The maximum number of characters to truncate each text to before tokenizing it.
    Setting this too low can cause unwantedly aggressive truncation, but setting it
    not too high can speed up tokenization. For English texts, a value of 8 is safe
    because average token lengths are generally much less than 8 characters."""

    pair_query_id_col: str = "QUERY_ID"
    """Column name for query ids in the input data."""

    pair_document_id_col: str = "DOCUMENT_ID"
    """Column name for document ids in the input data."""

    pair_query_text_col: str = "QUERY_TEXT"
    """Column name for query text in the input data."""

    pair_document_text_col: str = "DOCUMENT_TEXT"
    """Column name for document text in the input data."""

    out_query_id_col: str = "BATCH_QUERY_ID"
    """Column name for query ids in the output data. IDs are guaranteed unique
    by batch, not globally."""

    out_document_id_col: str = "BATCH_DOCUMENT_ID"
    """Column name for document ids in the output data. IDs are guaranteed unique
    by batch, not globally."""

    out_query_tokens_col: str = "QUERY_TOKEN_ID_LIST"
    """Column name for tokenized queries in the output data."""

    out_document_tokens_col: str = "DOCUMENT_TOKEN_ID_LIST"
    """Column name for tokenized documents in the output data."""

    out_relevance_col: str = "RELEVANCE"
    """Column name for relevance label integer values in the output data. 1 indicates
    a positive query-document pair, and no other values will be used in pretokenizing
    pair data.
    """

    shard_info: ShardedJobInfo = ShardedJobInfo(shard_id=0, num_shards=1)
    """Optional configuration to split a large job into many smaller jobs."""

    def split_to_shards(self, num_shards: int) -> List[PretokenizeAndBatchPairsConfig]:
        """Split one big job into multiple smaller jobs for parallel execution."""
        if self.shard_info.num_shards > 1:
            raise ValueError("Can't split a config that already has multiple shards")
        res = []
        for shard_id in range(num_shards):
            shard_info = ShardedJobInfo(shard_id=shard_id, num_shards=num_shards)
            res.append(self.model_copy(update={"shard_info": shard_info}))
        return res


class PretokenizeAndBatchMinedDataConfig(BaseModel):
    # Pydantic config.
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_default=True,
        use_attribute_docstrings=True,
        populate_by_name=True,
    )

    # Other fields.
    tokenizer_id: str
    """Unique identifier of a tokenizer on the huggingface hub."""

    query_loc: str
    """Location of the query text data in parquet format. Can be a single file
    or a directory of files."""

    document_loc: str
    """Location of the document text data in parquet format. Can be a single file
    or a directory of files."""

    labels_loc: str
    """Location of the relevance label data in parquet format. Can be a single file
    or a directory of files."""

    output_loc: str
    """Directory to write the output to."""

    query_prefix: str
    """Prefix to add to queries before tokenizing. Can be the empty string."""

    document_prefix: str
    """Prefix to add to documents before tokenizing. Can be the empty string."""

    random_seed: int = 0
    """Random seed for shuffling the order of queries in the dataset."""

    queries_per_batch: int = 256
    """The number of queries to include in each batch of output data."""

    num_query_uses: int = 1
    """The number of times to reuse a query (with different negatives) in
    different batches."""

    num_negatives_per_query_per_batch: int = 10
    """The number of negative documents to include for each query in the
    input data. If a query has more than this number of documents per use of the
    query, the most relevant negatives will be used (and randomly split up between
    uses of the query, if `num_query_uses > 1`)."""

    max_pos_per_query: int = 1
    """The maximum number of positive examples to select per query. A random
    selection is made, possibly allowing reuse across repeated uses of queries."""

    max_seq_length: int = 1024
    """The maximum sequence length to truncate to during pretokenization."""

    pre_truncate_max_chars_per_token: int = 8
    """The maximum number of characters to truncate each text to before tokenizing it.
    Setting this too low can cause unwantedly aggressive truncation, but setting it
    not too high can speed up tokenization. For English texts, a value of 8 is safe
    because average token lengths are generally much less than 8 characters."""

    in_query_id_col: str = "QUERY_ID"
    """Column name for query ids in the input data."""

    in_document_id_col: str = "DOCUMENT_ID"
    """Column name for document ids in the input data."""

    in_query_text_col: str = "QUERY_TEXT"
    """Column name for query text in the input data."""

    in_document_text_col: str = "DOCUMENT_TEXT"
    """Column name for document text in the input data."""

    in_label_relation_col: str = "RELEVANCE"
    """Column name for query-document relevance in the input data. Any numerical
    data type is accepted, and higher is considered more relevant."""

    out_query_id_col: str = "BATCH_QUERY_ID"
    """Column name for query ids in the output data. IDs are guaranteed unique
    by batch, not globally."""

    out_document_id_col: str = "BATCH_DOCUMENT_ID"
    """Column name for document ids in the output data. IDs are guaranteed unique
    by batch, not globally."""

    out_query_tokens_col: str = "QUERY_TOKEN_ID_LIST"
    """Column name for tokenized queries in the output data."""

    out_document_tokens_col: str = "DOCUMENT_TOKEN_ID_LIST"
    """Column name for tokenized documents in the output data."""

    out_relevance_col: str = "RELEVANCE"
    """Column name for relevance label integer values in the output data. 1 indicates
    a positive query-document pair, and no other values will be used in pretokenizing
    pair data.
    """


def _ith_random_subset_idx(total_count: int, subset_size: int, total_splits: int, split_i: int) -> NDArray[np.int64]:
    assert split_i < total_splits, f"{split_i=} {total_splits=}"
    total_taken = subset_size * total_splits
    assert total_taken <= total_count, "Requested more items than available"
    rand_idx = np.random.default_rng(seed=0).permutation(total_count)[:total_taken]
    return np.split(rand_idx, total_splits)[split_i]


def pretokenize_and_batch_query_doc_pairs(
    config: PretokenizeAndBatchPairsConfig,
) -> None:
    """Pre-tokenize and batch pairs of queries and documents for largescale pretraining.

    Input data is expected to be in parquet format as a directory containing one
    or more files. The inputs files must have columns for query and document ids
    as well as the text of the queries and documents.

    The output is a directory containing subdirectories of batched pretokenized
    data in the standard Arctic Embed training data format.

    Input and output data locations may be local or remote.

    NOTE: Due to deduplication on query and document ids, each batch of output
    may end up with fewer rows of data than the specified batch size if the input
    data contains duplicate queries or documents (e.g. multiple documents for the
    same query).
    """

    logger.info(str(config))

    # Load the tokenizer.
    tokenizer = Tokenizer.from_pretrained(config.tokenizer_id)
    tokenizer.no_padding()
    tokenizer.no_truncation()

    # Stream our shard's slice of the dataset (if the job is not sharded, this
    # will be the whole dataset).
    ds_in = ManyParquetDataset.from_root_dir(config.query_doc_pair_dir)
    shard_slice_start, shard_slice_end = identify_slice_for_shard(
        num_rows=ds_in.num_rows,
        chunk_size=config.batch_size,
        shard_id=config.shard_info.shard_id,
        num_shards=config.shard_info.num_shards,
    )
    batch_start = shard_slice_start // config.batch_size
    input_iter = stream_sliced_dataset_in_chunks(
        ds=ds_in,
        slice_start=shard_slice_start,
        slice_end=shard_slice_end,
        chunk_size=config.batch_size,
        columns_subset=[
            config.pair_query_id_col,
            config.pair_document_id_col,
            config.pair_query_text_col,
            config.pair_document_text_col,
        ],
    )

    # Write tokenization metadata.
    metadata_path = join(config.output_dir, "metadata.json")
    out_fs = get_fs_for_path(config.output_dir)
    out_fs.mkdir(config.output_dir, create_parents=True)
    if out_fs.exists(metadata_path):
        logger.warning(
            f"Metadata file {metadata_path} already exists. This is normal in a sharded job and thus will be ignored."
        )
    else:
        dataset_metadata = {
            "tokenizer_huggingface_id": config.tokenizer_id,
            "batch_size": config.batch_size,
            "max_seq_length": config.max_seq_length,
            "query_prefix": config.query_prefix,
            "document_prefix": config.document_prefix,
        }
        out_fs.write_text(path=metadata_path, value=json.dumps(dataset_metadata, indent=2))

    # Go batch by batch, tokenizing and writing to parquet.
    for i, table in enumerate(input_iter):
        # Handle missing ids by using a hash of the text.
        qid = table[config.pair_query_id_col].to_numpy()
        did = table[config.pair_document_id_col].to_numpy()
        queries = table[config.pair_query_text_col].to_numpy()
        documents = table[config.pair_document_text_col].to_numpy()
        qid_missing = pd.isna(qid)
        if qid_missing.any():
            logger.info(f"Missing {qid_missing.sum()} query ids in batch {i}, filling in")
            text_hash_for_id_missing_queries = pd.util.hash_array(queries[qid_missing])
            qid[qid_missing] = "__MISSING_ID__" + text_hash_for_id_missing_queries
        did_missing = pd.isna(did)
        if did_missing.any():
            logger.info(f"Missing {did_missing.sum()} document ids in batch {i}, filling in")
            text_hash_for_id_missing_docs = pd.util.hash_array(documents[did_missing])
            did[did_missing] = "__MISSING_ID__" + text_hash_for_id_missing_docs

        # Dedupe queries and documents by id.
        qid_unique, qid_unique_idx = np.unique(qid, return_index=True)
        did_unique, did_unique_idx = np.unique(did, return_index=True)
        queries_unique = queries[qid_unique_idx]
        docs_unique = documents[did_unique_idx]

        # Map string-typed ids to integers for faster reads of just the ids.
        qid_map = {qid: i for i, qid in enumerate(qid_unique)}
        did_map = {did: i for i, did in enumerate(did_unique)}
        qid_batch = np.array([qid_map[x] for x in qid], dtype=np.int32)
        did_batch = np.array([did_map[x] for x in did], dtype=np.int32)
        qid_unique_batch = np.arange(len(qid_unique), dtype=np.int32)
        did_unique_batch = np.arange(len(did_unique), dtype=np.int32)
        table_qid = pa.table(
            {
                config.pair_query_id_col: qid_unique,
                config.out_query_id_col: qid_unique_batch,
            }
        )
        table_did = pa.table(
            {
                config.pair_document_id_col: did_unique,
                config.out_document_id_col: did_unique_batch,
            }
        )

        # Construct relevance table (all pairs are relevant to one another).
        table_relevance = pa.table(
            {
                config.out_query_id_col: qid_batch,
                config.out_document_id_col: did_batch,
                config.out_relevance_col: np.ones(len(table)),
            }
        )

        # Add prefixes to queries and docs.
        query_texts = [config.query_prefix + q for q in queries_unique]
        doc_texts = [config.document_prefix + d for d in docs_unique]

        # Tokenize unique queries and docs.
        query_tokens = tokenize_to_arrow(
            tokenizer=tokenizer,
            texts=query_texts,
            max_length=config.max_seq_length,
            pre_truncate_max_chars_per_token=config.pre_truncate_max_chars_per_token,
        )
        doc_tokens = tokenize_to_arrow(
            tokenizer=tokenizer,
            texts=doc_texts,
            max_length=config.max_seq_length,
            pre_truncate_max_chars_per_token=config.pre_truncate_max_chars_per_token,
        )

        # Create output tables.
        table_query_tokens = pa.table(
            {
                config.out_query_id_col: qid_unique_batch,
                config.out_query_tokens_col: query_tokens,
            }
        )
        table_doc_tokens = pa.table(
            {
                config.out_document_id_col: did_unique_batch,
                config.out_document_tokens_col: doc_tokens,
            }
        )

        # Write output tables.
        batch_i = batch_start + i
        output_dir = join(config.output_dir, f"batch_{batch_i:08d}")
        relevance_path = join(output_dir, "relations.parquet")
        query_tokens_path = join(output_dir, "queries.parquet")
        doc_tokens_path = join(output_dir, "documents.parquet")
        qid_path = join(output_dir, "qid_map.parquet")
        did_path = join(output_dir, "did_map.parquet")
        logger.info(f"Writing batch {batch_i} to {output_dir}")
        out_fs.mkdir(output_dir, create_parents=True)
        pq.write_table(table_qid, qid_path, filesystem=out_fs)
        pq.write_table(table_did, did_path, filesystem=out_fs)
        pq.write_table(table_relevance, relevance_path, filesystem=out_fs)
        pq.write_table(table_query_tokens, query_tokens_path, filesystem=out_fs)
        pq.write_table(table_doc_tokens, doc_tokens_path, filesystem=out_fs)

    logger.info(f"Job complete, results written to {config.output_dir}")


def pretokenize_and_batch_mined_data(
    config: PretokenizeAndBatchMinedDataConfig,
) -> None:
    """Pre-tokenize and batch data which is the output of hard negative mining.

    Input data is expected to be in parquet format either as single files or
    directories containing a sequence of parquet files.

    The output is a directory containing subdirectories of batched pretokenized
    data in the standard Arctic Embed training data format.

    Input and output data locations may be local or remote.

    NOTE: Due to deduplication on query and document ids and the possibility of
    mined negative id lists possibly containing fewer than the target number of
    negatives per query, each batch of output may end up with fewer rows of data
    than the targeted size. Batches will not exceed targeted size, however.
    """

    logger.info(str(config))

    # Seed the RNG.
    rng = np.random.default_rng(config.random_seed)

    # Load the tokenizer.
    tokenizer = Tokenizer.from_pretrained(config.tokenizer_id)
    tokenizer.no_padding()
    tokenizer.no_truncation()

    # Load all queries and documents into memory, pre-truncating to mitigate
    # memory impact.
    truncate_text_max_characters = config.max_seq_length * config.pre_truncate_max_chars_per_token
    logger.info(f"Loading queries from {config.query_loc}")
    queries = load_id_to_text_map_from_parquet(
        path=config.query_loc,
        id_col=config.in_query_id_col,
        text_col=config.in_query_text_col,
        truncate_text_max_chars=truncate_text_max_characters,
    )
    logger.info(f"Loading documents from {config.document_loc}")
    documents = load_id_to_text_map_from_parquet(
        path=config.document_loc,
        id_col=config.in_document_id_col,
        text_col=config.in_document_text_col,
        truncate_text_max_chars=truncate_text_max_characters,
    )

    # Load all relevance labels into memory as a pandas DataFrame so we can use
    # the gropuby API to group the labels by query.
    logger.info(f"Loading relevance labels from {config.labels_loc}")
    df_labels = pd.read_parquet(config.labels_loc)
    qid_to_df_labels_idx = df_labels.groupby(config.in_query_id_col).indices

    # Write tokenization metadata.
    metadata_path = join(config.output_loc, "metadata.json")
    data_path = join(config.output_loc, "data")
    out_fs = get_fs_for_path(config.output_loc)
    out_fs.mkdirs(data_path, exist_ok=True)
    dataset_metadata = {
        "tokenizer_huggingface_id": config.tokenizer_id,
        "queries_per_batch": config.queries_per_batch,
        "num_query_uses": config.num_query_uses,
        "max_negatives_per_query_per_batch": config.num_negatives_per_query_per_batch,
        "max_seq_length": config.max_seq_length,
        "query_prefix": config.query_prefix,
        "document_prefix": config.document_prefix,
    }
    logger.info(f"Writing tokenizer metadata to {metadata_path}")
    out_fs.write_text(path=metadata_path, value=json.dumps(dataset_metadata, indent=2))

    # Determine batch by query ids.
    all_qids = df_labels[config.in_query_id_col].astype(np.uint64).unique()
    num_batches_per_query_repeat = len(all_qids) // config.queries_per_batch
    assert num_batches_per_query_repeat > 0, "Too few queries for batch size"

    with tqdm(
        total=config.num_query_uses * num_batches_per_query_repeat,
        desc="tokenizing and batching",
        unit="batch",
    ) as pbar:
        for q_reuse in range(config.num_query_uses):
            # Get a new order for the queries for each reuse of the queries.
            rng.shuffle(all_qids)
            qid_batches = np.split(
                all_qids[: num_batches_per_query_repeat * config.queries_per_batch],
                num_batches_per_query_repeat,
            )

            # Go batch by batch through the queries.
            for batch_i, qid_batch in enumerate(qid_batches):
                batch_i += q_reuse * num_batches_per_query_repeat

                # Sample a positive doc and a set of negative docs for each query.
                qids = []
                did_chunks = []
                relevances = []
                for qid in qid_batch:
                    labels_idx = qid_to_df_labels_idx[qid]
                    df_labels_for_query = df_labels.iloc[labels_idx]
                    is_positive = df_labels_for_query[config.in_label_relation_col] > 0
                    positive_idx = np.where(is_positive)[0]
                    negative_idx = np.where(~is_positive)[0]
                    assert len(positive_idx) > 0, f"Query {qid} has no positive docs"
                    assert len(negative_idx) > 0, f"Query {qid} has no negative docs"
                    selected_positive_idx = rng.choice(
                        positive_idx,
                        size=min(len(positive_idx), config.max_pos_per_query),
                        replace=False,
                    )
                    neg_subset_idx = _ith_random_subset_idx(
                        total_count=len(negative_idx),
                        subset_size=config.num_negatives_per_query_per_batch,
                        total_splits=config.num_query_uses,
                        split_i=q_reuse,
                    )
                    selected_negative_idx = negative_idx[neg_subset_idx]
                    del neg_subset_idx
                    selected_idx = np.concatenate([selected_positive_idx, selected_negative_idx])
                    df_labels_for_query_selected = df_labels_for_query.iloc[selected_idx]
                    qids.extend([qid] * len(selected_idx))
                    did_chunks.append(
                        df_labels_for_query_selected[config.in_document_id_col].to_numpy(dtype=np.uint64)
                    )
                    relevances.extend(df_labels_for_query_selected[config.in_label_relation_col])
                dids = np.concatenate(did_chunks)
                del did_chunks

                # Construct the label table for this batch.
                table_relevance = pa.table(
                    {
                        config.out_query_id_col: pa.array(qids, type=pa.uint64()),
                        config.out_document_id_col: pa.array(dids, type=pa.uint64()),
                        config.out_relevance_col: relevances,
                    }
                )

                # Tokenize the queries and documents for this batch.
                # NOTE: We only tokenize the unique queries and documents -- duplicate
                # ids only occur in the rows of the *labels* table.
                unique_qids = qid_batch
                unique_dids = np.unique(dids)
                unique_query_texts = queries.loc[unique_qids].to_numpy()
                unique_doc_texts = documents.loc[unique_dids].to_numpy()
                unique_query_texts = [config.query_prefix + q for q in unique_query_texts]
                unique_doc_texts = [config.document_prefix + d for d in unique_doc_texts]
                pretruncate_chars_per_token = config.pre_truncate_max_chars_per_token
                query_tokens = tokenize_to_arrow(
                    tokenizer=tokenizer,
                    texts=unique_query_texts,
                    max_length=config.max_seq_length,
                    pre_truncate_max_chars_per_token=pretruncate_chars_per_token,
                )
                doc_tokens = tokenize_to_arrow(
                    tokenizer=tokenizer,
                    texts=unique_doc_texts,
                    max_length=config.max_seq_length,
                    pre_truncate_max_chars_per_token=pretruncate_chars_per_token,
                )

                # Construct the query and doc tables for this batch.
                table_query_tokens = pa.table(
                    {
                        config.out_query_id_col: unique_qids,
                        config.out_query_tokens_col: query_tokens,
                    }
                )
                table_doc_tokens = pa.table(
                    {
                        config.out_document_id_col: unique_dids,
                        config.out_document_tokens_col: doc_tokens,
                    }
                )

                # Write output tables.
                output_dir = join(data_path, f"batch_{batch_i:08d}")
                relevance_path = join(output_dir, "relations.parquet")
                query_tokens_path = join(output_dir, "queries.parquet")
                doc_tokens_path = join(output_dir, "documents.parquet")
                logger.info(f"Writing batch {batch_i} to {output_dir}")
                out_fs.mkdir(output_dir, exist_ok=True, create_parents=True)
                pq.write_table(table_relevance, relevance_path, filesystem=out_fs)
                pq.write_table(table_query_tokens, query_tokens_path, filesystem=out_fs)
                pq.write_table(table_doc_tokens, doc_tokens_path, filesystem=out_fs)

                # Update progress bar.
                pbar.update()

    logger.info(f"Job complete, results written to {config.output_loc}")


def load_id_to_text_map_from_parquet(
    path: str, id_col: str, text_col: str, truncate_text_max_chars: int | None = None
) -> pd.Series:
    """Load id-text pairs from a parquet file or directory of parquet files as
    a pandas series with the index being the ids.

    NOTE: A Pandas series is used to accommodate uint64-typed ids. Native Python
    integers and native Python dictionaries can cause key-value lookup issues.
    """
    # Create a ManyParquetDataset.
    fs = get_fs_for_path(path)
    if fs.isdir(path):
        ds = ManyParquetDataset.from_root_dir(path)
    else:
        ds = ManyParquetDataset.from_filepaths([path])

    # Stream tables chunk-by-chunk, truncating and updating the mapping.
    all_ids = []
    all_texts = []
    for table in ds.stream_tables(columns_subset=[id_col, text_col]):
        ids = table.column(id_col)
        texts = table.column(text_col)
        if truncate_text_max_chars is not None:
            texts = pc.utf8_slice_codeunits(texts, start=0, stop=truncate_text_max_chars)
        all_ids.append(ids.to_numpy())
        all_texts.append(texts.to_numpy())

    all_ids_array = np.concatenate(all_ids)
    all_texts_array = np.concatenate(all_texts)
    return pd.Series(all_texts_array, index=all_ids_array)


def identify_slice_for_shard(num_rows: int, chunk_size: int, shard_id: int, num_shards: int) -> tuple[int, int]:
    # Figure out how to split the dataset into shards where the size of each shard
    # is divizible by chunks size (dropping the last sub-chunk-size set of rows).
    if chunk_size * num_shards > num_rows:
        msg = f"Too many shards and too large chunk size {num_rows=} {chunk_size=} {num_shards=}"
        raise ValueError(msg)
    base_shard_size = num_rows // num_shards
    base_shard_size -= base_shard_size % chunk_size
    if base_shard_size == 0:
        raise ValueError(
            f"Sharding dataset with {num_rows} rows into {num_shards} shards "
            f"is not possible at chunk size {chunk_size}"
        )
    remainder = num_rows - base_shard_size * num_shards
    remainder_batches = remainder // chunk_size
    assert remainder_batches < num_shards
    starts = []
    start = 0
    for i in range(num_shards):
        starts.append(start)
        start += base_shard_size
        if i < remainder_batches:
            start += chunk_size
    assert starts[-1] + base_shard_size == num_rows - (num_rows % chunk_size)
    start = starts[shard_id]
    end = start + base_shard_size if shard_id == num_shards - 1 else starts[shard_id + 1]
    shard_size = end - start
    assert shard_size % chunk_size == 0
    logger.info(
        f"[{shard_id=} {num_shards=}] Streaming {shard_size / chunk_size:,} "
        f"chunks of {chunk_size:,} rows from row {start:,} to row {end:,}"
    )

    return start, end


def stream_sliced_dataset_in_chunks(
    ds: ManyParquetDataset,
    slice_start: int,
    slice_end: int,
    chunk_size: int,
    columns_subset: list[str] | None = None,
) -> Iterator[pa.Table]:
    # Figure out which files to read, and which leading/trailing rows to drop.
    # NOTE: Binary search feels less readable than a linear scan here.
    rows_skipped_by_skipped_files = 0
    for first_file_i, file_rows in enumerate(ds.num_rows_per_file):
        if rows_skipped_by_skipped_files + file_rows > slice_start:
            break
        rows_skipped_by_skipped_files += file_rows
    rows_covered_after_last_file = 0
    for last_file_i, file_rows in enumerate(ds.num_rows_per_file):
        rows_covered_after_last_file += file_rows
        if rows_covered_after_last_file > slice_end:
            break
    assert last_file_i >= first_file_i
    skip_first_k = slice_start - rows_skipped_by_skipped_files
    skip_last_k = rows_covered_after_last_file - slice_end

    # Slice the dataset to the indicated files.
    ds_slice = ManyParquetDataset(
        num_rows_per_file=ds.num_rows_per_file[first_file_i : last_file_i + 1],
        pq_filepaths=ds.pq_filepaths[first_file_i : last_file_i + 1],
    )

    # Chunked streaming.
    saved: list[pa.Table] = []
    with tqdm(
        total=slice_end - slice_start,
        unit="row",
        unit_scale=True,
        desc="streaming dataset",
    ) as pbar:
        for i, table in enumerate(ds_slice.stream_tables(columns_subset=columns_subset)):
            if i == 0:
                table = table.slice(offset=skip_first_k)
            if i == ds_slice.num_files - 1:
                table = table.slice(length=len(table) - skip_last_k)
            saved.append(table)
            while sum(map(len, saved)) >= chunk_size:
                combined = pa.concat_tables(saved)
                out = combined.slice(length=chunk_size)
                saved = [combined.slice(offset=chunk_size)]
                pbar.update(len(out))
                yield out


def tokenize_to_arrow(
    tokenizer: Tokenizer,
    texts: List[str],
    max_length: Optional[int] = None,
    pre_truncate_max_chars_per_token: Optional[int] = 8,
) -> pa.LargeListArray:
    """Tokenize a batch of text data into an Arrow LargeListArray.

    NOTE: This is not threadsafe due to modifying the tokenizer's state.
    """
    if tokenizer.padding is not None:
        raise ValueError("Provided tokenizer must be set to `no_padding` mode.")

    # Determine token datatype.
    vocab_size = tokenizer.get_vocab_size()
    token_dtype: Type[np.integer] = np.uint16  # type: ignore[type-arg]
    if vocab_size > np.iinfo(token_dtype).max:
        token_dtype = np.uint32

    # Run truncated tokenization.
    prior_truncation = tokenizer.truncation
    if max_length is not None:
        # NOTE: Not threadsafe!
        tokenizer.enable_truncation(max_length=max_length)

        # Pre-truncate by character length to bound wasted computation in very
        # long strings.
        if pre_truncate_max_chars_per_token is not None:
            texts = [t[: pre_truncate_max_chars_per_token * max_length] for t in texts]
    encodings = tokenizer.encode_batch(texts)
    if prior_truncation is None:
        tokenizer.no_truncation()
    else:
        tokenizer.enable_truncation(**prior_truncation)

    # Pack the tokens into Arrow data.
    item_lengths = []
    all_ids = []
    for encoding in encodings:
        ids = encoding.ids
        item_lengths.append(len(ids))
        all_ids.extend(ids)
    # NOTE: This can be the slowest part of the entire function by far.
    # In testing, about 2/3 of tokenization time was spent just cleaning up these lists.
    del encodings
    all_ids_array = np.asarray(all_ids, dtype=token_dtype)
    del all_ids
    item_lengths.insert(0, 0)
    offsets_arary = np.asarray(item_lengths).cumsum()
    tokens_array = pa.LargeListArray.from_arrays(offsets=offsets_arary, values=all_ids_array)

    return tokens_array
