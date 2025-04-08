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
from tempfile import TemporaryDirectory

import fire
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from arctic_embed.data_processing.dense_retrieval import dense_retrieval
from arctic_embed.data_processing.utils import normalize_embeddings
from arctic_embed.data_processing.utils import read_embeddings_from_pq
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def main(
    query_pq_dir: str,
    doc_pq_dir: str,
    out_path: str | Path,
    retrieval_depth: int = 300,
    # NOTE: For large datasets, we will run out of memory loading everything onto
    # GPU, so instead we have to swap slices of query and item vectors onto and off
    # of the GPU. The following parameters control the size of those slices, and you
    # should reduce them if you run out of GPU memory running this script.
    item_slice_size: int = 1024000,  # Max number of doc vectors on GPU at one time.
    query_slice_size: int = 4096,  # Max number of query vectors on GPU at one time.
) -> None:
    """CLI entrypoint for performing dense retrieval to mine for negatives.

    For more sophisticated use cases, write your own script that implements
    something like this main function but adapted to your needs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    with TemporaryDirectory(dir=out_path.parent) as tempdir:
        memmap_file_path_q = Path(tempdir) / "memmap_q.bin"
        memmap_file_path_d = Path(tempdir) / "memmap_d.bin"
        query_ids, query_embeddings = read_embeddings_from_pq(
            sorted(Path(query_pq_dir).glob("*.parquet")),
            vector_memmap_file=memmap_file_path_q,
        )
        doc_ids, doc_embeddings = read_embeddings_from_pq(
            sorted(Path(doc_pq_dir).glob("*.parquet")),
            vector_memmap_file=memmap_file_path_d,
        )
        normalize_embeddings(query_embeddings, inplace=True)
        normalize_embeddings(doc_embeddings, inplace=True)
        result_indices, result_scores = dense_retrieval(
            query_embeddings=query_embeddings,
            item_embeddings=doc_embeddings,
            retrieval_depth=retrieval_depth,
            item_slice_size=item_slice_size,
            query_slice_size=query_slice_size,
        )

    # Write the scores to disk chunk by chunk.
    print("Writing retrieval results to disk")
    out_schema = pa.schema(
        {
            "QUERY_ID": pa.infer_type(query_ids),
            "DOCUMENT_ID": pa.infer_type(doc_ids),
            "SCORE": pa.float32(),
        }
    )
    with pq.ParquetWriter(out_path, out_schema) as pq_writer, tqdm(total=result_indices.size, unit="score") as pbar:
        chunk_size = 1024
        for slice_start in range(0, result_indices.shape[0], chunk_size):
            slice_end = slice_start + chunk_size
            slice_table = pa.table(
                {
                    "QUERY_ID": np.repeat(query_ids[slice_start:slice_end], retrieval_depth),
                    "DOCUMENT_ID": doc_ids[result_indices[slice_start:slice_end].ravel()],
                    "SCORE": result_scores[slice_start:slice_end].ravel(),
                }
            )
            pq_writer.write_table(slice_table)
            pbar.update(len(slice_table))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    with logging_redirect_tqdm():
        fire.Fire(main)
