"""Re-batch pretokenized Arctic Embed datasets to a uniform query batch size.

This utility reads one or more existing pretokenized datasets (either on local
storage or S3), shuffles their batches, and rewrites them so that every output
batch contains the same number of queries. Documents and relevance labels are
preserved for each query. The datasets are assumed to share a tokenizer and the
same tokenization configuration (prefixes, max sequence lengths, etc.).

Example usage:

```
python rebatch_pretokenized.py \
  --input-roots s3://bucket/ds_a s3://bucket/ds_b \
  --output-root s3://bucket/ds_mixed \
  --queries-per-batch 512 \
  --shuffle-seed 123
```

NOTE: This script keeps the per-query set of positives/negatives exactly as
they appeared in the source batches. If different datasets use different counts
of hard negatives (e.g. 10 vs. 30), the output batches will contain the mixed
set of counts.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from typing import Dict
from typing import Iterable
from typing import List
from typing import Sequence
from typing import Tuple

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs
from tqdm.auto import tqdm


QUERY_TOKEN_COLUMN = "QUERY_TOKEN_ID_LIST"
DOC_TOKEN_COLUMN = "DOCUMENT_TOKEN_ID_LIST"
QUERY_BATCH_ID_COLUMN = "BATCH_QUERY_ID"
DOC_BATCH_ID_COLUMN = "BATCH_DOCUMENT_ID"
RELATION_VALUE_COLUMN = "RELEVANCE"


class QueryExample:
    """In-memory representation of a query and its associated documents."""

    __slots__ = ("query_tokens", "doc_examples")

    def __init__(self, query_tokens: Sequence[int], doc_examples: List[Tuple[Sequence[int], int]]):
        self.query_tokens = query_tokens
        self.doc_examples = doc_examples


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebatch pretokenized datasets to a uniform batch size.")
    parser.add_argument(
        "--input-roots",
        nargs="+",
        required=True,
        help="List of dataset roots (local paths or s3:// URIs) each containing data/batch_*/",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Destination root (local path or s3:// URI) for the rebatched dataset.",
    )
    parser.add_argument(
        "--queries-per-batch",
        type=int,
        required=True,
        help="Number of queries per output batch.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Seed used to shuffle source batches before reprocessing (default: 0).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a progress bar while converting batches.",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=4,
        help="Number of concurrent workers used to prefetch batches (default: 4).",
    )
    return parser.parse_args()


def list_batch_directories(fs: fsspec.AbstractFileSystem, root_path: str) -> List[str]:
    entries = fs.ls(root_path, detail=True)
    batch_dirs = [entry["name"] for entry in entries if entry.get("type") == "directory"]
    batch_dirs.sort()
    if len(batch_dirs) == 0:
        raise ValueError(f"No batch directories found under {root_path}")
    return batch_dirs


def read_batch_tables(
    fs: fsspec.AbstractFileSystem,
    batch_dir: str,
) -> Tuple[pa.Table, pa.Table, pa.Table]:
    queries = pq.read_table(str(PurePosixPath(batch_dir) / "queries.parquet"), filesystem=fs)
    documents = pq.read_table(str(PurePosixPath(batch_dir) / "documents.parquet"), filesystem=fs)
    relations = pq.read_table(str(PurePosixPath(batch_dir) / "relations.parquet"), filesystem=fs)
    return queries, documents, relations


def extract_query_examples(
    queries_table: pa.Table,
    documents_table: pa.Table,
    relations_table: pa.Table,
) -> Iterable[QueryExample]:
    query_ids = queries_table.column(QUERY_BATCH_ID_COLUMN).to_numpy(zero_copy_only=False)
    query_token_lists = queries_table.column(QUERY_TOKEN_COLUMN).to_pylist()

    document_ids = documents_table.column(DOC_BATCH_ID_COLUMN).to_numpy(zero_copy_only=False)
    document_token_lists = documents_table.column(DOC_TOKEN_COLUMN).to_pylist()
    doc_id_to_tokens = {int(doc_id): doc_tokens for doc_id, doc_tokens in zip(document_ids, document_token_lists)}

    relations_by_query: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    rel_q_ids = relations_table.column(QUERY_BATCH_ID_COLUMN).to_numpy(zero_copy_only=False)
    rel_d_ids = relations_table.column(DOC_BATCH_ID_COLUMN).to_numpy(zero_copy_only=False)
    rel_values = relations_table.column(RELATION_VALUE_COLUMN).to_numpy(zero_copy_only=False)
    for qid, did, rel in zip(rel_q_ids, rel_d_ids, rel_values):
        rel_int = int(rel)
        if rel_int == 0:
            rel_int = -1
        relations_by_query[int(qid)].append((int(did), rel_int))

    for qid, q_tokens in zip(query_ids, query_token_lists):
        doc_examples = []
        for did, rel in relations_by_query[int(qid)]:
            doc_tokens = doc_id_to_tokens.get(did)
            if doc_tokens is None:
                raise KeyError(f"Missing document tokens for document id {did}")
            doc_examples.append((doc_tokens, rel))
        if not doc_examples:
            raise ValueError("Encountered query with zero associated documents; this should not happen.")
        yield QueryExample(q_tokens, doc_examples)


def build_large_list_array(token_lists: Sequence[Sequence[int]], value_type: pa.DataType) -> pa.LargeListArray:
    offsets = [0]
    flat_values: List[int] = []
    for tokens in token_lists:
        offsets.append(offsets[-1] + len(tokens))
        flat_values.extend(tokens)
    offsets_array = pa.array(offsets, type=pa.int64())
    values_array = pa.array(flat_values, type=value_type)
    return pa.LargeListArray.from_arrays(offsets_array, values_array)


def write_batch(
    fs: fsspec.AbstractFileSystem,
    output_root: str,
    batch_index: int,
    examples: Sequence[QueryExample],
    query_id_type: pa.DataType,
    doc_id_type: pa.DataType,
    relation_value_type: pa.DataType,
    query_token_value_type: pa.DataType,
    doc_token_value_type: pa.DataType,
) -> None:
    batch_dir = PurePosixPath(output_root) / f"batch_{batch_index:08d}"
    fs.makedirs(str(batch_dir), exist_ok=True)

    query_token_lists: List[Sequence[int]] = []
    doc_token_lists: List[Sequence[int]] = []
    relations_q: List[int] = []
    relations_d: List[int] = []
    relations_v: List[int] = []

    doc_tokens_to_index: Dict[Tuple[int, ...], int] = {}

    for q_idx, example in enumerate(examples):
        query_token_lists.append(example.query_tokens)
        for doc_tokens, rel in example.doc_examples:
            doc_key = tuple(doc_tokens)
            doc_idx = doc_tokens_to_index.get(doc_key)
            if doc_idx is None:
                doc_idx = len(doc_token_lists)
                doc_tokens_to_index[doc_key] = doc_idx
                doc_token_lists.append(doc_tokens)
            relations_q.append(q_idx)
            relations_d.append(doc_idx)
            relations_v.append(rel)

    query_ids_array = pa.array(np.arange(len(query_token_lists)), type=query_id_type)
    doc_ids_array = pa.array(np.arange(len(doc_token_lists)), type=doc_id_type)

    queries_table = pa.table(
        {
            QUERY_BATCH_ID_COLUMN: query_ids_array,
            QUERY_TOKEN_COLUMN: build_large_list_array(query_token_lists, query_token_value_type),
        }
    )

    documents_table = pa.table(
        {
            DOC_BATCH_ID_COLUMN: doc_ids_array,
            DOC_TOKEN_COLUMN: build_large_list_array(doc_token_lists, doc_token_value_type),
        }
    )

    relations_table = pa.table(
        {
            QUERY_BATCH_ID_COLUMN: pa.array(relations_q, type=query_id_type),
            DOC_BATCH_ID_COLUMN: pa.array(relations_d, type=doc_id_type),
            RELATION_VALUE_COLUMN: pa.array(relations_v, type=relation_value_type),
        }
    )

    pq.write_table(queries_table, str(batch_dir / "queries.parquet"), filesystem=fs)
    pq.write_table(documents_table, str(batch_dir / "documents.parquet"), filesystem=fs)
    pq.write_table(relations_table, str(batch_dir / "relations.parquet"), filesystem=fs)


def rebatch_datasets(args: argparse.Namespace) -> None:
    dataset_specs = []
    for root in args.input_roots:
        fs, path = url_to_fs(root)
        batch_dirs = list_batch_directories(fs, path)
        dataset_specs.append((fs, batch_dirs))

    # Determine types from the first batch of the first dataset.
    sample_fs, sample_batch_dirs = dataset_specs[0]
    sample_queries, sample_docs, sample_relations = read_batch_tables(sample_fs, sample_batch_dirs[0])
    query_id_type = sample_queries.schema.field(QUERY_BATCH_ID_COLUMN).type
    doc_id_type = sample_docs.schema.field(DOC_BATCH_ID_COLUMN).type
    relation_value_type = sample_relations.schema.field(RELATION_VALUE_COLUMN).type
    query_token_value_type = sample_queries.schema.field(QUERY_TOKEN_COLUMN).type.value_type
    doc_token_value_type = sample_docs.schema.field(DOC_TOKEN_COLUMN).type.value_type

    # Prepare output filesystem and write metadata.
    out_fs, out_path = url_to_fs(args.output_root)
    out_fs.makedirs(out_path, exist_ok=True)

    # Enumerate all batch directories across datasets and optionally shuffle them.
    all_batches: List[Tuple[fsspec.AbstractFileSystem, str]] = []
    for fs, batch_dirs in dataset_specs:
        all_batches.extend((fs, batch_dir) for batch_dir in batch_dirs)

    rng = random.Random(args.shuffle_seed)
    rng.shuffle(all_batches)

    pending_examples: List[QueryExample] = []
    batch_index = 0
    total_batches = len(all_batches)
    if total_batches == 0:
        raise ValueError("No batch directories found across provided inputs.")

    progress = tqdm(total=total_batches, desc="Rebatching", disable=not args.progress)

    max_workers = max(1, args.prefetch_workers)
    batch_iter = iter(all_batches)
    pending_futures = deque()

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            fs, batch_dir = next(batch_iter)
        except StopIteration:
            return False
        future = executor.submit(read_batch_tables, fs, batch_dir)
        pending_futures.append((future, fs, batch_dir))
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(min(max_workers, total_batches)):
            submit_next(executor)

        while pending_futures:
            future, fs, batch_dir = pending_futures.popleft()
            queries_table, documents_table, relations_table = future.result()
            submit_next(executor)

            for example in extract_query_examples(queries_table, documents_table, relations_table):
                pending_examples.append(example)
                if len(pending_examples) == args.queries_per_batch:
                    write_batch(
                        out_fs,
                        out_path,
                        batch_index,
                        pending_examples,
                        query_id_type,
                        doc_id_type,
                        relation_value_type,
                        query_token_value_type,
                        doc_token_value_type,
                    )
                    pending_examples = []
                    batch_index += 1

            progress.update(1)

    if pending_examples:
        write_batch(
            out_fs,
            out_path,
            batch_index,
            pending_examples,
            query_id_type,
            doc_id_type,
            relation_value_type,
            query_token_value_type,
            doc_token_value_type,
        )
        batch_index += 1

    progress.close()
    print(f"Wrote {batch_index} batches to {args.output_root}")


def main() -> None:
    args = parse_arguments()
    rebatch_datasets(args)


if __name__ == "__main__":
    main()

