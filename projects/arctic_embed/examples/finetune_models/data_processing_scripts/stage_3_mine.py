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
import pandas as pd
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
    df_relevance_scores = pd.read_parquet(relevance_score_pq_path)
    df_labels = pd.read_parquet(labels_pq_path)
    qid_to_label_idx = df_labels.groupby("QUERY_ID").indices
    qid_to_score_idx = df_relevance_scores.groupby("QUERY_ID").indices
    union_query_ids = sorted(set(qid_to_label_idx.keys()) & set(qid_to_score_idx.keys()))
    label_docid_array = df_labels["DOCUMENT_ID"].to_numpy()
    score_docid_array = df_relevance_scores["DOCUMENT_ID"].to_numpy()
    score_value_array = df_relevance_scores["SCORE"].to_numpy()

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
            # of the current query.
            score_idx = qid_to_score_idx[query_id]
            doc_ids = score_docid_array[score_idx]
            scores = score_value_array[score_idx]
            del score_idx

            # Find which of these scores are positive.
            pos_doc_ids = label_docid_array[qid_to_label_idx[query_id]]
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

    # Write the mined relevances to disk chunk by chunk.
    out_schema = pa.schema(
        {
            "QUERY_ID": pa.infer_type(df_labels["QUERY_ID"]),
            "DOCUMENT_ID": pa.infer_type(df_labels["DOCUMENT_ID"]),
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
