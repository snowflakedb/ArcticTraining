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
from typing import Iterator

import fire
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


def main(
    relevance_score_pq_path: str,
    labels_pq_path: str,
    out_path: str | Path,
    max_negative_to_positive_relevance_threshold: float = 0.95,
    negative_samples_per_query: int = 10,
    max_positives_per_query: int = 1,
) -> None:
    """CLI entrypoint for mining retrieval results from relevance scores.

    For more sophisticated use cases, write your own script that implements
    something like this main function but adapted to your needs.
    """
    assert str(out_path).endswith(".parquet")

    # Load data and organize dense retrieval scores and annotated labels by query id.
    #
    # Memory note: the scores parquet can be enormous (triviaqa is ~2.7B rows of
    # uint64/uint64/float32). The previous implementation held, at once: the full scores
    # pandas DataFrame + numpy copies of its columns + a `groupby().indices` dict with one
    # position-array per query (~2.5x the raw data => OOM-killed the 160GiB job). Instead:
    #   * read each column straight to numpy via pyarrow (zero-copy, no DataFrame doubling),
    #   * group by query WITHOUT a global sort -- dense-retrieval output is already written
    #     grouped per query, so we detect contiguous runs in O(n) and address each query's
    #     rows with a cheap (start, end) span (slices are VIEWS); we fall back to a stable
    #     argsort only if the input is not already grouped,
    #   * free each query-id array as soon as its spans are built.
    # The per-query mining math below is unchanged, so the output is identical.
    def _col_to_numpy(path: str, col: str, dtype=None) -> np.ndarray:
        table = pq.read_table(path, columns=[col])
        arr = table.column(col).combine_chunks().to_numpy(zero_copy_only=False)
        del table
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype, copy=False)
        return arr

    def _is_grouped(qid_array: np.ndarray) -> bool:
        """True if every query id occupies a single contiguous run (no global sort needed).
        Only inspects the value at each run boundary, so it is cheap (~#queries, not #rows)."""
        if len(qid_array) <= 1:
            return True
        run_start = np.empty(len(qid_array), dtype=bool)
        run_start[0] = True
        np.not_equal(qid_array[1:], qid_array[:-1], out=run_start[1:])
        run_values = qid_array[run_start]
        return len(run_values) == len(set(run_values.tolist()))

    def _spans(qid_array: np.ndarray) -> tuple[np.ndarray, dict]:
        """Unique qids and a {qid: (start, end)} map for a qid array whose equal values are
        contiguous. O(n), one bool mask -- no internal sort (unlike np.unique)."""
        if len(qid_array) == 0:
            return qid_array[:0], {}
        run_start = np.empty(len(qid_array), dtype=bool)
        run_start[0] = True
        np.not_equal(qid_array[1:], qid_array[:-1], out=run_start[1:])
        starts = np.flatnonzero(run_start)
        ends = np.append(starts[1:], len(qid_array))
        uniq = qid_array[starts]
        return uniq, dict(zip(uniq.tolist(), zip(starts.tolist(), ends.tolist())))

    def _load_grouped(path: str, is_labels: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Read (QUERY_ID, DOCUMENT_ID[, SCORE]) and return (doc_ids, scores_or_None,
        uniq_qids, qid->span), grouping rows by query id (stable, no global sort unless
        the file is not already grouped). The labels file has no SCORE column."""
        qids = _col_to_numpy(path, "QUERY_ID")
        dids = _col_to_numpy(path, "DOCUMENT_ID")
        scores = None if is_labels else _col_to_numpy(path, "SCORE", np.float32)
        if not _is_grouped(qids):
            order = np.argsort(qids, kind="stable")
            qids = qids[order]
            dids = dids[order]
            if scores is not None:
                scores = scores[order]
            del order
        uniq, spans = _spans(qids)
        del qids
        return dids, scores, uniq, spans

    # Scores: also capture the output id arrow types from this (the larger) file.
    _score_schema = pq.read_schema(relevance_score_pq_path)
    out_qid_arrow_type = _score_schema.field("QUERY_ID").type
    out_did_arrow_type = _score_schema.field("DOCUMENT_ID").type
    score_docid_array, score_value_array, score_qids, qid_to_score_idx = _load_grouped(relevance_score_pq_path)
    label_docid_array, _, label_qids, qid_to_label_idx = _load_grouped(labels_pq_path, is_labels=True)
    union_query_ids = sorted(set(label_qids.tolist()) & set(score_qids.tolist()))

    # Create the output directory.
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    # Log the job details.
    logger.info(
        f"Mining negatives from {relevance_score_pq_path} and {labels_pq_path} "
        f"to {out_path}. {max_negative_to_positive_relevance_threshold=} "
        f"{negative_samples_per_query=}"
    )

    # Go through all query ids and mine negatives which have the highest relevance
    # scores below the threshold. Yield (query id, doc id, relevance) relations in
    # in chunks.
    def iter_mined_relevances(
        min_chunk_size: int = 10_000,
    ) -> Iterator[tuple[list[int], list[int], list[int]]]:
        chunk_qids = []
        chunk_dids = []
        chunk_relations = []
        skip_count: int = 0
        drop_pos_count: int = 0
        for query_id in tqdm(union_query_ids, unit="query"):
            # Slice to the scores of all docs and the annotated relevance labels
            # of the current query. Spans are contiguous (data was sorted by QUERY_ID),
            # so these slices are views — no per-query copies.
            score_start, score_end = qid_to_score_idx[query_id]
            doc_ids = score_docid_array[score_start:score_end]
            scores = score_value_array[score_start:score_end]

            # Find which of these scores are positive.
            label_start, label_end = qid_to_label_idx[query_id]
            pos_doc_ids = label_docid_array[label_start:label_end]
            is_pos_doc = np.isin(element=doc_ids, test_elements=pos_doc_ids)
            del pos_doc_ids

            # Optimization: Finish fast if we have no labeled positives.
            if np.sum(is_pos_doc) == 0:
                logger.debug(
                    f"None of the labeled positive docs have scores in the top {len(doc_ids):,} scores. Skipping."
                )
                skip_count += 1
                continue

            # Select the highest scoring postive documents to use.
            pos_scores = scores[is_pos_doc]
            idx_pos_score_sort = np.argsort(pos_scores)[::-1]
            sorted_pos_ids = doc_ids[is_pos_doc][idx_pos_score_sort]
            use_pos_ids = sorted_pos_ids[:max_positives_per_query]
            drop_pos_count += len(sorted_pos_ids[max_positives_per_query:])

            # Use minimum score of used positives as the score for thresholding.
            min_score_position = min(len(idx_pos_score_sort), max_positives_per_query) - 1
            pos_score = pos_scores[idx_pos_score_sort[min_score_position]]
            cutoff = max_negative_to_positive_relevance_threshold * pos_score

            # Apply the false negative cutoff and select hardest eligible negatives.
            is_low_false_negative_risk = scores < cutoff
            is_neg_eligible = is_low_false_negative_risk & (~is_pos_doc)
            neg_eligible_scores = scores[is_neg_eligible]
            idx_neg_score_sort = np.argsort(neg_eligible_scores)[::-1]
            negative_doc_ids = doc_ids[is_neg_eligible][idx_neg_score_sort][:negative_samples_per_query]
            if len(negative_doc_ids) < negative_samples_per_query:
                logger.debug(
                    f"Query {query_id} has fewer than {negative_samples_per_query} negative samples. Skipping"
                )
                skip_count += 1
                continue

            chunk_qids.extend([query_id] * (len(use_pos_ids) + len(negative_doc_ids)))
            chunk_dids.extend(use_pos_ids)
            chunk_dids.extend(negative_doc_ids)
            chunk_relations.extend([1] * len(use_pos_ids))
            chunk_relations.extend([-1] * len(negative_doc_ids))

            if len(chunk_qids) >= min_chunk_size:
                yield chunk_qids, chunk_dids, chunk_relations
                chunk_qids.clear()
                chunk_dids.clear()
                chunk_relations.clear()

        # Yield the leftover chunk.
        if len(chunk_qids) > 0:
            yield chunk_qids, chunk_dids, chunk_relations

        if skip_count > 0:
            logger.warning(f"Dropped {skip_count:,}/{len(union_query_ids):,} queries due to false negative risk.")
        if drop_pos_count > 0:
            logger.warning(
                f"Dropped {drop_pos_count:,} positive documents because we were limited to {max_positives_per_query=}"
            )

    # Write the mined relevances to disk chunk by chunk. (ID arrow types were captured
    # from the scores frame before it was freed; QUERY_ID/DOCUMENT_ID share a dtype
    # across the scores and labels inputs.)
    out_schema = pa.schema(
        {
            "QUERY_ID": out_qid_arrow_type,
            "DOCUMENT_ID": out_did_arrow_type,
            "RELEVANCE": pa.int8(),
        }
    )
    with pq.ParquetWriter(out_path, out_schema) as pq_writer:
        for chunk_qids, chunk_dids, chunk_relations in iter_mined_relevances():
            chunk_col_map = {
                "QUERY_ID": chunk_qids,
                "DOCUMENT_ID": chunk_dids,
                "RELEVANCE": chunk_relations,
            }
            chunk_table = pa.table(chunk_col_map, schema=out_schema)
            pq_writer.write_table(chunk_table)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    with logging_redirect_tqdm():
        fire.Fire(main)
