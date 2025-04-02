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
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Iterable
from typing import Iterator
from typing import Sequence

import fire
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from arctic_embed.data_processing.teacher_models import AbstractEmbedder
from arctic_embed.data_processing.teacher_models import Arctic2LargeEmbedder
from arctic_embed.data_processing.utils import NDArrayOfFloat
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)


def count_rows_in_parquet_files(filepaths: Sequence[str]) -> int:
    return sum(pq.ParquetFile(fp).metadata.num_rows for fp in tqdm(filepaths, unit="file", desc="counting rows"))


def stream_id_text_pairs_from_parquet_files(
    filepaths: Sequence[str], id_column_name: str, text_column_name: str
) -> Iterator[tuple[int, str]]:
    for filepath in filepaths:
        table = pq.read_table(filepath, columns=[id_column_name, text_column_name])
        for record_batch in table.to_batches():
            ids = record_batch.column(id_column_name).to_pylist()
            texts = record_batch.column(text_column_name).to_pylist()
            for result in zip(ids, texts):
                yield result


def embed_texts_to_parquet_files(
    embedder: AbstractEmbedder,
    id_text_pairs: Iterable[tuple[int, str]],
    is_query: bool,
    out_dir: str | Path,
    batch_size: int,
    id_column_name: str = "ID",
    vector_column_name: str = "VECTOR",
    pbar_total: int | None = None,
    max_row_per_file: int = 512_000,
) -> None:
    assert batch_size < max_row_per_file

    logger.info(f"Embedding data to {out_dir} using {embedder.__class__.__name__}. {batch_size=}, {max_row_per_file=}")

    # Get an embedder for every available GPU.
    num_gpu = torch.cuda.device_count()
    if num_gpu > 0:
        logger.info(f"Copying embedder to all {num_gpu} GPUs")
        device_embedders = []
        for device_id in range(num_gpu):
            device = torch.device(device_id)
            device_embedder = deepcopy(embedder)
            device_embedder.to_device(device)
            device_embedders.append(device_embedder)
    else:
        device_embedders = [embedder]

    def _iter_batches() -> Iterator[tuple[Sequence[int], Sequence[str]]]:
        batch_ids = []
        batch_texts = []
        for id, text in id_text_pairs:
            batch_ids.append(id)
            batch_texts.append(text)
            if len(batch_ids) == batch_size:
                yield batch_ids, batch_texts
                batch_ids = []
                batch_texts = []
        if len(batch_ids) > 0:
            yield batch_ids, batch_texts

    def _embed_on_device(
        args: tuple[int, tuple[Sequence[int], Sequence[str]]],
    ) -> tuple[Sequence[int], NDArrayOfFloat]:
        i, (batch_ids, batch_texts) = args
        device_id = i % num_gpu
        device_embedder = device_embedders[device_id]
        return batch_ids, device_embedder.embed_batch(batch_texts, is_query=is_query)

    pq_writer: pq.ParquetWriter | None = None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_part_count = -1  # Start at -1 because we increment before using.
    current_file_rowcount = 0
    try:
        num_devices = len(device_embedders)
        with ThreadPool(num_devices) as embed_pool, tqdm(
            total=pbar_total, unit="text", desc="embedding texts to parquet"
        ) as pbar:
            # Round robin devices to balance load as we iterate batches and embed
            # the batches in parallel via multi-threading.
            id_vector_batch_iter = embed_pool.imap(_embed_on_device, enumerate(_iter_batches()))

            for batch_ids, batch_vectors in id_vector_batch_iter:
                # Ensure ids are uint64.
                batch_ids = pa.array(batch_ids, type=pa.uint64())

                # Construct an Arrow Table for the embeddings.
                chunk_table = _construct_embedding_table(
                    ids=batch_ids,
                    vectors=batch_vectors,
                    id_column_name=id_column_name,
                    vector_column_name=vector_column_name,
                )

                # Write the table to the Parquet file.
                if pq_writer is None:
                    file_part_count += 1
                    out_path = out_dir / f"part_{file_part_count:04d}.parquet"
                    pq_writer = pq.ParquetWriter(out_path, chunk_table.schema)
                if current_file_rowcount + len(chunk_table) > max_row_per_file:
                    file_part_count += 1
                    out_path = out_dir / f"part_{file_part_count:04d}.parquet"
                    pq_writer.close()
                    pq_writer = pq.ParquetWriter(out_path, chunk_table.schema)
                    current_file_rowcount = 0
                pq_writer.write_table(chunk_table)
                current_file_rowcount += len(chunk_table)

                # Update the progress bar.
                pbar.update(len(chunk_table))
    finally:
        if pq_writer is not None:
            pq_writer.close()


def _construct_embedding_table(
    ids: Sequence[int],
    vectors: NDArrayOfFloat,
    id_column_name: str,
    vector_column_name: str,
) -> pa.Table:
    assert vectors.ndim == 2
    assert vectors.shape[0] == len(ids)
    vectors_array = pa.FixedSizeListArray.from_arrays(vectors.ravel(), list_size=vectors.shape[1])
    return pa.table({id_column_name: ids, vector_column_name: vectors_array})


def simple_cli(
    query_pq_path: str,
    doc_pq_path: str,
    out_dir: str | Path,
    batch_size: int = 32,
    max_row_per_file: int = 512_000,
    max_seq_len: int = 512,
) -> None:
    """Simple CLI entrypoint for embedding query and document text from
    specifically-formatted parquet files using the Arctic 2 Large model.

    For more sophisticated use cases, write your own script that implements
    something like this main function but adapted to your needs.
    """
    out_dir = Path(out_dir)
    embedder = Arctic2LargeEmbedder(max_seq_len=max_seq_len)
    num_q = count_rows_in_parquet_files([query_pq_path])
    id_text_pairs_q = stream_id_text_pairs_from_parquet_files(
        [query_pq_path], id_column_name="QUERY_ID", text_column_name="QUERY_TEXT"
    )
    logger.info("Embedding queries")
    embed_texts_to_parquet_files(
        embedder,
        id_text_pairs_q,
        is_query=True,
        out_dir=out_dir / "query_embeddings",
        batch_size=batch_size,
        max_row_per_file=max_row_per_file,
        pbar_total=num_q,
    )
    num_d = count_rows_in_parquet_files([doc_pq_path])
    id_text_pairs_d = stream_id_text_pairs_from_parquet_files(
        [doc_pq_path], id_column_name="DOCUMENT_ID", text_column_name="DOCUMENT_TEXT"
    )
    logger.info("Embedding documents")
    embed_texts_to_parquet_files(
        embedder,
        id_text_pairs_d,
        is_query=False,
        out_dir=out_dir / "document_embeddings",
        batch_size=batch_size,
        max_row_per_file=max_row_per_file,
        pbar_total=num_d,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    with logging_redirect_tqdm():
        fire.Fire(simple_cli)
