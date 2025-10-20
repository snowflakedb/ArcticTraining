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

from __future__ import annotations

import json
import logging
from multiprocessing.pool import AsyncResult
from multiprocessing.pool import ThreadPool
from os.path import dirname
from os.path import join
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import Optional

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import IterableDataset

from .slice_sparse_coo_tensor import slice_sparse_coo_tensor

logger = logging.getLogger(__name__)


COLNAME_QUERY_TOKENS = "QUERY_TOKEN_ID_LIST"
COLNAME_DOCUMENT_TOKENS = "DOCUMENT_TOKEN_ID_LIST"
COLNAME_QUERY_BATCH_ID = "BATCH_QUERY_ID"
COLNAME_DOCUMENT_BATCH_ID = "BATCH_DOCUMENT_ID"
COLNAME_RELEVANCE = "RELEVANCE"


class CollatedTokens(NamedTuple):
    padded_ids: NDArray[np.int64]
    seq_lens: NDArray[np.int64]
    pad_value: int
    is_left_padded: bool


class ContrastiveLearningBatch(NamedTuple):
    query_tokens: Tensor
    query_attention_mask: Tensor
    document_tokens: Tensor
    document_attention_mask: Tensor
    relevance_labels: Tensor  # NOTE: This should be a sparse COO tensor!

    def to_device(self, device: torch.device, non_blocking: bool = False) -> ContrastiveLearningBatch:
        return ContrastiveLearningBatch(
            query_tokens=self.query_tokens.to(device, non_blocking=non_blocking),
            query_attention_mask=self.query_attention_mask.to(device, non_blocking=non_blocking),
            document_tokens=self.document_tokens.to(device, non_blocking=non_blocking),
            document_attention_mask=self.document_attention_mask.to(device, non_blocking=non_blocking),
            relevance_labels=self.relevance_labels.to(device, non_blocking=non_blocking),
        )


class ContrastiveLearningBatchDataset(IterableDataset[ContrastiveLearningBatch]):
    def __init__(
        self,
        filesystem: fsspec.AbstractFileSystem,
        root_directory: str,
        split_factor: int = 1,
        shard_id: int = 0,
        world_size: int = 1,
        pad_value: int = 0,
        left_pad: bool = False,
        max_seq_len_query: Optional[int] = None,
        max_seq_len_doc: Optional[int] = None,
        device: Optional[torch.device] = None,
        start_batch_idx: int = 0,  # Add support for resuming from specific batch
        preserve_relations_on_split: bool = False,
    ) -> None:
        super().__init__()
        self.filesystem = filesystem
        self.root_directory = root_directory
        self.split_factor = split_factor
        self.shard_id = shard_id
        self.world_size = world_size
        self.pad_value = pad_value
        self.left_pad = left_pad
        self.max_seq_len_query = max_seq_len_query
        self.max_seq_len_doc = max_seq_len_doc
        self.device = device
        self.start_batch_idx = start_batch_idx
        self.preserve_relations_on_split = preserve_relations_on_split

        # Look up the batch directories.
        batch_paths = sorted(filesystem.ls(root_directory))
        assert len(batch_paths) > 0, f"No batches subdirectories in {root_directory=}"
        self.batch_paths = batch_paths
        logger.info(
            "[dataset.init]"
            f" shard_id={self.shard_id} world_size={self.world_size} split_factor={self.split_factor} root={self.root_directory}"
        )

    def __len__(self) -> int:
        return len(self.split_factor * self.batch_paths)

    # NOTE: This dataset could have been a regular dataset, but this causes a lot of
    # inefficiency when reading in order with split_factor > 1 due to batches
    # being loaded and split multiple times, so instead we make this an
    # IterableDataset to avoid inefficient data access patterns.
    # def __getitem__(self, index: int) -> ContrastiveLearningBatch:
    #     batch_dir_i, split_i = divmod(index, self.split_factor)
    #     batch = read_batch(
    #         self.batch_paths[batch_dir_i],
    #         filesystem=self.filesystem,
    #         pad_value=self.pad_value,
    #         left_pad=self.left_pad,
    #     )
    #     batch = split_batch(batch, self.split_factor)[split_i]
    #     batch = shard_batch(batch, self.shard_id, self.world_size)
    #     if self.device is not None:
    #         batch = batch.to_device(self.device, non_blocking=True)
    #     return batch

    def _read_batches_for_path(
        self,
        batch_directory: str,
    ) -> List[ContrastiveLearningBatch]:
        """Helper function to perform all splitting, sharding, and device
        movement for the batch(es) contained in a single directory.
        """
        un_split_batch = read_batch(
            batch_directory=batch_directory,
            filesystem=self.filesystem,
            pad_value=self.pad_value,
            left_pad=self.left_pad,
            max_seq_len_query=self.max_seq_len_query,
            max_seq_len_doc=self.max_seq_len_doc,
        )
        if self.preserve_relations_on_split:
            split_batches = split_batch_preserve_relations(un_split_batch, self.split_factor)
        else:
            split_batches = split_batch(un_split_batch, self.split_factor)

        # Skip split batches that would result in zero-length shards for the current
        # world size. This happens when the (already split) batch has fewer samples
        # than the number of ranks, leading to empty tensors downstream.
        filtered_splits: List[ContrastiveLearningBatch] = []
        for split_idx, split_batch_item in enumerate(split_batches):
            num_queries = split_batch_item.query_tokens.size(0)
            num_documents = split_batch_item.document_tokens.size(0)
            if num_queries < self.world_size or num_documents < self.world_size:
                logger.warning(
                    "[dataset.batch] Skipping split %d from %s because it has "
                    "%d queries and %d documents, which is fewer than world_size=%d",
                    split_idx,
                    batch_directory,
                    num_queries,
                    num_documents,
                    self.world_size,
                )
                continue
            filtered_splits.append(split_batch_item)

        if not filtered_splits:
            logger.warning(
                "[dataset.batch] Skipping entire batch directory %s because"
                " all %d splits were smaller than world_size=%d",
                batch_directory,
                len(split_batches),
                self.world_size,
            )
            return []

        split_sharded_batches = [shard_batch(b, self.shard_id, self.world_size) for b in filtered_splits]
        # Move only the first split batch to the device, to avoid hogging device memory.
        if self.device is not None and len(split_sharded_batches) > 0:
            split_sharded_batches[0] = split_sharded_batches[0].to_device(self.device, non_blocking=True)
            b0 = split_sharded_batches[0]
            logger.info(
                f"[dataset.batch] dir={batch_directory} "
                f"q_tokens={b0.query_tokens.size(0)} d_tokens={b0.document_tokens.size(0)} "
                f"shard_id={self.shard_id}/{self.world_size} split_factor={self.split_factor}"
            )
        return split_sharded_batches

    def __iter__(self) -> Iterator[ContrastiveLearningBatch]:
        # Try reading the metadata file to get the tokenization metadata.
        root_parent = dirname(self.root_directory.rstrip("/"))
        metadata_path = join(root_parent, "metadata.json")
        if self.filesystem.exists(metadata_path):
            meta_json = self.filesystem.read_text(metadata_path)
            tokenization_metadata = json.loads(meta_json)
        else:
            # Try reading one more parent directory up for metadata.
            root_parent_parent = dirname(root_parent)
            metadata_path = join(root_parent_parent, "metadata.json")
            if self.filesystem.exists(metadata_path):
                meta_json = self.filesystem.read_text(metadata_path)
                tokenization_metadata = json.loads(meta_json)
            # Fall back to not displaying metadata.
            else:
                tokenization_metadata = "<no tokenization metadata found>"
        logger.info(f"Iterating dataset:  {self.root_directory} | {tokenization_metadata}")

        # Track the current step index
        step_idx = 0

        # Calculate which batches to skip based on start_batch_idx (which is actually start_step_idx)
        batches_to_skip = 0
        remaining_steps_to_skip = 0

        if self.start_batch_idx > 0:
            # start_batch_idx is actually the step index we want to start from
            batches_to_skip = self.start_batch_idx // self.split_factor
            remaining_steps_to_skip = self.start_batch_idx % self.split_factor

            logger.info(f"Will skip first {self.start_batch_idx} steps to resume from step {self.start_batch_idx}")
            logger.info(
                f"Skipping {batches_to_skip} complete batch files (each with {self.split_factor} steps), "
                f"and {remaining_steps_to_skip} additional steps in the next batch"
            )

            # Skip the appropriate number of batch paths
            batch_paths_to_use = self.batch_paths[batches_to_skip:]
            # Update step counter to reflect skipped steps
            step_idx = batches_to_skip * self.split_factor
        else:
            batch_paths_to_use = self.batch_paths

        path_iter = iter(batch_paths_to_use)
        first_path = next(path_iter, None)

        # Edge case: no batches.
        if first_path is None:
            return

        # Set up a pool of a single thread for concurrent work.
        with ThreadPool(1) as pool:
            future: Optional[AsyncResult] = pool.apply_async(func=self._read_batches_for_path, args=(first_path,))

            # Loop until we've loaded everything (and have no future).
            while future is not None:
                # Start reading one path ahead to overlap IO and compute with GPU
                # work being done if there is a training step being run on each batch
                # we emit.
                next_path = next(path_iter, None)
                if next_path is not None:
                    next_future: Optional[AsyncResult] = pool.apply_async(
                        func=self._read_batches_for_path,
                        args=(next_path,),
                    )
                else:
                    next_future = None

                # Yield the split sharded batches from the last future, pushing
                # each next sharded split batch to device non-blocking as we go.
                split_sharded_batches = future.get()

                # For the first batch after skipping complete batches, we may need to skip some steps
                if step_idx == batches_to_skip * self.split_factor and remaining_steps_to_skip > 0:
                    # Skip the remaining steps in the first batch
                    split_sharded_batches = split_sharded_batches[remaining_steps_to_skip:]
                    step_idx += remaining_steps_to_skip

                if len(split_sharded_batches) > 0:
                    ssb = split_sharded_batches[0]
                    for next_ssb in split_sharded_batches[1:]:
                        if self.device is not None:
                            next_ssb = next_ssb.to_device(self.device, non_blocking=True)

                        yield ssb
                        step_idx += 1

                        ssb = next_ssb

                    # Handle the last batch
                    yield ssb
                    step_idx += 1

                # Move up to the next future.
                future = next_future


def read_dataset_metadata(root_directory: str, filesystem: fsspec.AbstractFileSystem) -> Dict[str, Any]:
    meatadata_json = filesystem.read_text(join(root_directory, "metadata.json"))
    metadata = json.loads(meatadata_json)
    return metadata


def read_batch(
    batch_directory: str,
    filesystem: Optional[fsspec.AbstractFileSystem] = None,
    pad_value: int = 0,
    left_pad: bool = False,
    max_seq_len_query: Optional[int] = None,
    max_seq_len_doc: Optional[int] = None,
) -> ContrastiveLearningBatch:
    # Read the data from parquet concurrently.
    pq_path_query = join(batch_directory, "queries.parquet")
    pq_path_doc = join(batch_directory, "documents.parquet")
    pq_path_rel = join(batch_directory, "relations.parquet")
    pq_paths = (pq_path_query, pq_path_doc, pq_path_rel)
    with ThreadPool(3) as pool:
        t_query, t_doc, t_rel = pool.map(
            lambda kwd: pq.read_table(**kwd),
            (dict(source=path, filesystem=filesystem) for path in pq_paths),
        )

    # Ensure queries and documents are unique (by id) within the batch.
    qids = t_query.column(COLNAME_QUERY_BATCH_ID).to_numpy()
    dids = t_doc.column(COLNAME_DOCUMENT_BATCH_ID).to_numpy()
    assert len(np.unique(qids)) == len(qids), "Queries are not unique."
    assert len(np.unique(dids)) == len(dids), "Documents are not unique."

    # Collate the tokens.
    tokens_q = t_query.column(COLNAME_QUERY_TOKENS)
    tokens_d = t_doc.column(COLNAME_DOCUMENT_TOKENS)
    coll_q = collate_tokens(tokens_q, pad_value=pad_value, left_pad=left_pad, max_seq_len=max_seq_len_query)
    coll_d = collate_tokens(tokens_d, pad_value=pad_value, left_pad=left_pad, max_seq_len=max_seq_len_doc)

    # Create attention masks.
    attn_q = attn_from_lengths(coll_q.seq_lens, is_left_paddded=coll_q.is_left_padded)
    attn_d = attn_from_lengths(coll_d.seq_lens, is_left_paddded=coll_d.is_left_padded)

    # Create a matrix of relevance labels.
    qid_to_idx_map_ufunc = np.vectorize({qid: idx for idx, qid in enumerate(qids)}.get)
    did_to_idx_map_ufunc = np.vectorize({did: idx for idx, did in enumerate(dids)}.get)
    relations_qid = t_rel.column(COLNAME_QUERY_BATCH_ID).to_numpy()
    relations_did = t_rel.column(COLNAME_DOCUMENT_BATCH_ID).to_numpy()
    relations_values = t_rel.column(COLNAME_RELEVANCE).to_numpy()
    relation_q_idx = qid_to_idx_map_ufunc(relations_qid)
    relation_d_idx = did_to_idx_map_ufunc(relations_did)
    relation_matrix = torch.sparse_coo_tensor(
        indices=torch.tensor(np.vstack([relation_q_idx, relation_d_idx])),
        values=torch.tensor(relations_values),
        size=(len(qids), len(dids)),
    ).coalesce()

    # Return the collated batch.
    return ContrastiveLearningBatch(
        query_tokens=torch.from_numpy(coll_q.padded_ids),
        query_attention_mask=torch.from_numpy(attn_q),
        document_tokens=torch.from_numpy(coll_d.padded_ids),
        document_attention_mask=torch.from_numpy(attn_d),
        relevance_labels=relation_matrix,
    )


def shard_batch(batch: ContrastiveLearningBatch, shard_id: int, world_size: int) -> ContrastiveLearningBatch:
    """Shard a batch of data for distributed training."""
    assert 0 <= shard_id < world_size

    # Short circuit when no sharding is necessary.
    if world_size == 1:
        return batch

    # Shard the query and document tokens.
    q_split_size = batch.query_tokens.size(0) // world_size
    d_split_size = batch.document_tokens.size(0) // world_size
    q_start = q_split_size * shard_id
    q_end = q_start + q_split_size
    d_start = d_split_size * shard_id
    d_end = d_start + d_split_size
    query_tokens = batch.query_tokens[q_start:q_end, ...]
    query_attention_mask = batch.query_attention_mask[q_start:q_end, ...]
    document_tokens = batch.document_tokens[d_start:d_end, ...]
    document_attention_mask = batch.document_attention_mask[d_start:d_end, ...]

    # We pass the full relevance label matrix to all shards, but if we lost some
    # indivisible remaining number of query or document examples when sharding
    # to evenly-sized shards, we also need to drop the corresponding relevance
    # labels.
    trim_slices = [(None, q_split_size * world_size), (None, d_split_size * world_size)]
    relevance_labels = slice_sparse_coo_tensor(batch.relevance_labels, trim_slices)

    # Return the sharded batch.
    return ContrastiveLearningBatch(
        query_tokens=query_tokens,
        query_attention_mask=query_attention_mask,
        document_tokens=document_tokens,
        document_attention_mask=document_attention_mask,
        relevance_labels=relevance_labels,
    )


def split_batch(batch: ContrastiveLearningBatch, total_splits: int) -> List[ContrastiveLearningBatch]:
    """Split a larger batch into smaller ones.

    This is intended to support use-cases in which the data is pre-batched to a
    large size (e.g. 32k) and we want to train at a smaller batch size which evenly
    divides this larger size (e.g. 8k).
    """
    assert total_splits > 0
    if total_splits == 1:
        return [batch]

    split_batches = []
    q_split_size = batch.query_tokens.size(0) // total_splits
    d_split_size = batch.document_tokens.size(0) // total_splits
    for i in range(total_splits):
        # Figure out the slice indices.
        q_start = q_split_size * i
        q_end = q_start + q_split_size
        d_start = d_split_size * i
        d_end = d_start + d_split_size

        # Split the query and document tokens.
        query_tokens = batch.query_tokens[q_start:q_end, ...]
        query_attention_mask = batch.query_attention_mask[q_start:q_end, ...]
        document_tokens = batch.document_tokens[d_start:d_end, ...]
        document_attention_mask = batch.document_attention_mask[d_start:d_end, ...]

        # Slice the relevance labels via sparse matrix manipulation.
        relevance_labels = batch.relevance_labels.coalesce()
        sliced_relevance_labels = slice_sparse_coo_tensor(relevance_labels, [(q_start, q_end), (d_start, d_end)])

        # Construct the new split batch.
        split_batch = ContrastiveLearningBatch(
            query_tokens=query_tokens,
            query_attention_mask=query_attention_mask,
            document_tokens=document_tokens,
            document_attention_mask=document_attention_mask,
            relevance_labels=sliced_relevance_labels,
        )
        split_batches.append(split_batch)
    return split_batches


def split_batch_preserve_relations(batch: ContrastiveLearningBatch, total_splits: int) -> List[ContrastiveLearningBatch]:
    """Split a batch into smaller ones while keeping each query's labeled docs.

    Queries are grouped into equally-sized chunks (up to one-off differences due to
    integer division). For each chunk we gather the union of all documents that
    participate in a labeled relation with those queries, ensuring positives and
    hard negatives remain present after the split.
    """

    assert total_splits > 0
    if total_splits == 1:
        return [batch]

    num_queries = batch.query_tokens.size(0)
    num_documents = batch.document_tokens.size(0)

    if num_queries == 0 or num_documents == 0:
        return [batch]

    relations = batch.relevance_labels.coalesce()
    if relations._nnz() == 0:
        return split_batch(batch, total_splits)

    # Build query -> document adjacency (include positives and negatives alike).
    indices = relations.indices()
    query_to_docs: List[set[int]] = [set() for _ in range(num_queries)]
    for idx in range(indices.shape[1]):
        q_idx = int(indices[0, idx])
        d_idx = int(indices[1, idx])
        query_to_docs[q_idx].add(d_idx)

    if any(len(doc_set) == 0 for doc_set in query_to_docs):
        # Fall back if the dataset has queries without labeled docs.
        return split_batch(batch, total_splits)

    # Determine the desired query count per split.
    if total_splits >= num_queries:
        desired_sizes = [1] * num_queries
    else:
        base = num_queries // total_splits
        remainder = num_queries % total_splits
        desired_sizes = [base + (1 if split_idx < remainder else 0) for split_idx in range(total_splits)]
        desired_sizes = [size for size in desired_sizes if size > 0]
        # Ensure all queries are assigned even if integer math truncates.
        assigned = sum(desired_sizes)
        if assigned < num_queries:
            desired_sizes[-1] += num_queries - assigned

    split_batches: List[ContrastiveLearningBatch] = []
    query_cursor = 0

    def _ordered_doc_indices(doc_set: set[int]) -> List[int]:
        doc_mask = torch.zeros(num_documents, dtype=torch.bool)
        if doc_set:
            doc_mask[list(doc_set)] = True
        return torch.nonzero(doc_mask, as_tuple=False).squeeze(1).tolist()

    for desired_size in desired_sizes:
        if query_cursor >= num_queries:
            break
        q_end = min(query_cursor + desired_size, num_queries)
        q_indices = list(range(query_cursor, q_end))
        query_cursor = q_end

        doc_set: set[int] = set()
        for q_idx in q_indices:
            doc_set.update(query_to_docs[q_idx])

        doc_indices = _ordered_doc_indices(doc_set)
        if not doc_indices:
            return split_batch(batch, total_splits)

        q_idx_tensor = torch.tensor(q_indices, dtype=torch.long)
        d_idx_tensor = torch.tensor(doc_indices, dtype=torch.long)

        split_batches.append(
            ContrastiveLearningBatch(
                query_tokens=batch.query_tokens[q_idx_tensor],
                query_attention_mask=batch.query_attention_mask[q_idx_tensor],
                document_tokens=batch.document_tokens[d_idx_tensor],
                document_attention_mask=batch.document_attention_mask[d_idx_tensor],
                relevance_labels=_slice_relations_by_indices(relations, q_idx_tensor, d_idx_tensor),
            )
        )

    # Handle any remaining queries (e.g., when total_splits < num_queries).
    if query_cursor < num_queries:
        q_indices = list(range(query_cursor, num_queries))
        doc_set: set[int] = set()
        for q_idx in q_indices:
            doc_set.update(query_to_docs[q_idx])
        doc_indices = _ordered_doc_indices(doc_set)
        if not doc_indices:
            return split_batch(batch, total_splits)
        q_idx_tensor = torch.tensor(q_indices, dtype=torch.long)
        d_idx_tensor = torch.tensor(doc_indices, dtype=torch.long)
        split_batches.append(
            ContrastiveLearningBatch(
                query_tokens=batch.query_tokens[q_idx_tensor],
                query_attention_mask=batch.query_attention_mask[q_idx_tensor],
                document_tokens=batch.document_tokens[d_idx_tensor],
                document_attention_mask=batch.document_attention_mask[d_idx_tensor],
                relevance_labels=_slice_relations_by_indices(relations, q_idx_tensor, d_idx_tensor),
            )
        )

    total_queries_in_splits = sum(sb.query_tokens.size(0) for sb in split_batches)
    # We expect roughly ``total_splits`` splits unless the dataset has fewer queries
    # than required. Fall back if we did not produce the intended number and the
    # remainder of queries is non-zero (indicating we had to abort early).
    if (total_queries_in_splits != num_queries) or (len(split_batches) and len(split_batches) < min(total_splits, num_queries)):
        print("Fallback to split_batch")
        return split_batch(batch, total_splits)

    return split_batches


def _connected_components_from_relations(relations: torch.Tensor) -> List[Tuple[List[int], List[int]]]:
    relations = relations.coalesce()
    if relations._nnz() == 0:
        return []

    indices = relations.indices()
    query_idx = indices[0].tolist()
    doc_idx = indices[1].tolist()

    num_queries = relations.size(0)
    num_docs = relations.size(1)

    query_to_docs: Dict[int, set[int]] = defaultdict(set)
    doc_to_queries: Dict[int, set[int]] = defaultdict(set)
    for q, d in zip(query_idx, doc_idx):
        query_to_docs[q].add(d)
        doc_to_queries[d].add(q)

    visited_queries = [False] * num_queries
    visited_docs = [False] * num_docs
    components: List[Tuple[List[int], List[int]]] = []

    for seed_query in range(num_queries):
        if visited_queries[seed_query] or seed_query not in query_to_docs:
            continue
        q_component: List[int] = []
        d_component: List[int] = []
        queue: deque[Tuple[str, int]] = deque()
        queue.append(("q", seed_query))
        visited_queries[seed_query] = True

        while queue:
            kind, idx = queue.popleft()
            if kind == "q":
                q_component.append(idx)
                for doc in query_to_docs.get(idx, ()):  # type: ignore[arg-type]
                    if not visited_docs[doc]:
                        visited_docs[doc] = True
                        queue.append(("d", doc))
            else:
                d_component.append(idx)
                for query in doc_to_queries.get(idx, ()):  # type: ignore[arg-type]
                    if not visited_queries[query]:
                        visited_queries[query] = True
                        queue.append(("q", query))

        components.append((sorted(q_component), sorted(d_component)))

    return components


def _slice_relations_by_indices(relations: torch.Tensor, query_indices: torch.Tensor, doc_indices: torch.Tensor) -> torch.Tensor:
    relations = relations.coalesce()
    device = relations.device
    query_indices = query_indices.to(device)
    doc_indices = doc_indices.to(device)

    num_queries = relations.size(0)
    num_docs = relations.size(1)

    q_map = torch.full((num_queries,), -1, dtype=torch.long, device=device)
    d_map = torch.full((num_docs,), -1, dtype=torch.long, device=device)
    q_map[query_indices] = torch.arange(query_indices.size(0), device=device)
    d_map[doc_indices] = torch.arange(doc_indices.size(0), device=device)

    indices = relations.indices()
    values = relations.values()

    mapped_q = q_map[indices[0]]
    mapped_d = d_map[indices[1]]
    mask = (mapped_q >= 0) & (mapped_d >= 0)

    if mask.sum() == 0:
        empty_indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        empty_values = torch.empty((0,), dtype=values.dtype, device=device)
        return torch.sparse_coo_tensor(empty_indices, empty_values, size=(query_indices.size(0), doc_indices.size(0))).coalesce()

    new_indices = torch.stack([mapped_q[mask], mapped_d[mask]], dim=0)
    new_values = values[mask]
    return torch.sparse_coo_tensor(new_indices, new_values, size=(query_indices.size(0), doc_indices.size(0))).coalesce()


def collate_tokens(
    tokens: pa.LargeListArray,
    max_seq_len: Optional[int] = None,
    pad_value: int = 0,
    left_pad: bool = False,
) -> CollatedTokens:
    assert len(tokens) > 0

    # Convert from Arrow to an NDArray of object dtype containing other
    # NDArrays of tokens sequences (integer object type).
    # This makes things faster than Arrow does.
    tokens_array_of_arrays = tokens.to_numpy(zero_copy_only=False)

    # Figure out sequence lengths after truncation.
    seq_lens = np.vectorize(len)(tokens_array_of_arrays)
    if max_seq_len is not None:
        seq_lens = np.minimum(seq_lens, max_seq_len)
    max_seq_len = seq_lens.max()
    assert max_seq_len is not None  # For type-checking.

    # Initialize the output.
    batch_size = len(tokens_array_of_arrays)
    padded_token_ids = np.full((batch_size, max_seq_len), pad_value, dtype=np.int64)

    # Insert items into the output.
    for i, (seq_len, seq) in enumerate(zip(seq_lens, tokens_array_of_arrays)):
        seq_truncated = seq[:seq_len]
        if left_pad:
            padded_token_ids[i, -seq_len:] = seq_truncated
        else:
            padded_token_ids[i, :seq_len] = seq_truncated

    # Return all useful details.
    return CollatedTokens(
        padded_ids=padded_token_ids,
        seq_lens=seq_lens,
        pad_value=pad_value,
        is_left_padded=left_pad,
    )


def attn_from_lengths(sequence_lengths: NDArray[np.int64], is_left_paddded: bool = False) -> NDArray[np.int64]:
    """Create an bidirectional attention mask from sequence lengths."""
    batch_size = sequence_lengths.shape[0]
    max_seq_len = sequence_lengths.max()
    attention_mask = np.zeros((batch_size, max_seq_len), dtype=np.int64)
    for i, seq_len in enumerate(sequence_lengths):
        if is_left_paddded:
            attention_mask[i, -seq_len:] = 1
        else:
            attention_mask[i, :seq_len] = 1
    return attention_mask
