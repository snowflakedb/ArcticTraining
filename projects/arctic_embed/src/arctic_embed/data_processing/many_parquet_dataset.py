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

from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.pool import ApplyResult
from multiprocessing.pool import ThreadPool
from os.path import join
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import cast

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ManyParquetDataset:
    """Abstraction for a dataset stored in many parquet files in either local or
    remote filesystem/object storage."""

    num_rows_per_file: Sequence[int]
    pq_filepaths: Sequence[str]
    num_threads_for_streaming_reads = 20
    num_row_readahead_for_streaming_reads = 10_000

    def __repr__(self) -> str:
        # Shorten the number of paths included in the string representation.
        if self.num_files > 3:
            num_rows_excerpt = "[" + ", ".join(map(str, self.num_rows_per_file[:3])) + "...]"
            paths_excerpt = "[" + ", ".join(self.pq_filepaths[:3]) + "...]"
        else:
            num_rows_excerpt = str(self.num_rows_per_file)
            paths_excerpt = str(self.pq_filepaths)
        return f"ManyParquetDataset(num_rows_per_file={num_rows_excerpt}, pq_filepaths={paths_excerpt})"

    @property
    def num_rows(self) -> int:
        return sum(self.num_rows_per_file)

    @property
    def num_files(self) -> int:
        return len(self.pq_filepaths)

    def batches_per_epoch(self, batch_size: int) -> int:
        return self.num_rows // batch_size

    @classmethod
    def from_root_dir(cls, root_dir: str, num_thread: int = 30, progress_bar: bool = True) -> ManyParquetDataset:
        filesystem = get_fs_for_path(root_dir)
        pq_paths = sorted(filesystem.glob(join(root_dir, "*.parquet")))
        result = cls.from_filepaths(pq_filepaths=pq_paths, num_thread=num_thread, progress_bar=progress_bar)
        if len(result.pq_filepaths) == 0:
            raise ValueError(f"Found zero parquet files for root directory {root_dir}")
        return result

    @classmethod
    def from_filepaths(
        cls,
        pq_filepaths: Sequence[str],
        num_thread: int = 30,
        progress_bar: bool = True,
    ) -> ManyParquetDataset:
        with ThreadPool(num_thread) as pool:
            row_count_iter = pool.imap(num_rows_in_pq_file, pq_filepaths)
            row_count_iter = tqdm(
                row_count_iter,
                total=len(pq_filepaths),
                desc="Counting dataset rows",
                unit="file",
                disable=not progress_bar,
                leave=False,
            )
            row_counts = tuple(row_count_iter)
        return cls(num_rows_per_file=row_counts, pq_filepaths=pq_filepaths)

    def stream_tables(self, columns_subset: Optional[Sequence[str]] = None) -> Iterator[pa.Table]:
        """Stream dataset from the filesystem as arrow tables.

        NOTE: One table per file.
        """
        assert self.num_threads_for_streaming_reads > 0
        counts_to_verify = list(self.num_rows_per_file)
        futures: List[ApplyResult[pa.Table]] = []
        reading_ahead_rows = 0
        # NOTE: We reuse the filesystem to avoid reestablishing connections to
        # remote filesystems.
        filesystem = get_fs_for_path(self.pq_filepaths[0])
        with ThreadPool(self.num_threads_for_streaming_reads) as pool:
            for p, p_rows in zip(self.pq_filepaths, self.num_rows_per_file):
                # Enforce the read ahead limit, always allowing at least two reads
                # to ensure some amount of parallelism.
                while len(futures) > 1 and reading_ahead_rows + p_rows > self.num_row_readahead_for_streaming_reads:
                    table = futures.pop(0).get()
                    reading_ahead_rows -= len(table)
                    expect_num_rows = counts_to_verify.pop(0)
                    if len(table) != expect_num_rows:
                        raise RuntimeError(
                            "Fatal mismatch between metadata and actual data. "
                            f"Metadata indicated {expect_num_rows:,d} rows, "
                            f"but actually got {len(table):,d} rows"
                        )
                    yield table
                futures.append(
                    pool.apply_async(
                        pq.read_table,
                        kwds=dict(source=p, columns=columns_subset, filesystem=filesystem),
                    )
                )
                reading_ahead_rows += p_rows
            for future, expect_num_rows in zip(futures, counts_to_verify):
                table = future.get()
                if len(table) != expect_num_rows:
                    raise RuntimeError(
                        "Fatal mismatch between metadata and actual data. "
                        f"Metadata indicated {expect_num_rows:,d} rows, "
                        f"but actually got {len(table):,d} rows"
                    )
                yield table

    def read_all_as_table(
        self,
        columns_subset: Optional[Sequence[str]] = None,
        progress_bar: bool = True,
        progress_bar_tag: str = "",
    ) -> pa.Table:
        chunks = []
        pbar_desc = "reading from parquet files" if progress_bar_tag == "" else f"reading {progress_bar_tag}"
        with tqdm(
            total=self.num_rows,
            desc=pbar_desc,
            unit="row",
            unit_scale=True,
            disable=not progress_bar,
        ) as pbar:
            for table in self.stream_tables(columns_subset=columns_subset):
                chunks.append(table)
                pbar.update(len(table))
        return pa.concat_tables(chunks)


_fs_cache = {}


def get_fs_for_path(path: str) -> fsspec.AbstractFileSystem:
    """Filesystem access with caching for minimizing overhead of repeated access."""
    if "://" in path:
        protocol = path.split("://", maxsplit=1)[0]
    else:
        protocol = "file"

    if protocol == "file":
        if path not in _fs_cache:
            _fs_cache["file"] = fsspec.filesystem("file")
        return _fs_cache["file"]
    elif protocol == "s3":
        if path not in _fs_cache:
            larger_blocksize = 64 * (1024**2)  # 64 MiB vs. default of 5MiB
            _fs_cache["s3"] = fsspec.filesystem("s3", use_listings_cache=False, default_block_size=larger_blocksize)
        return _fs_cache["s3"]
    else:
        raise ValueError(f"Unrecognized protocol for path `{path}`")


def num_rows_in_pq_file(pq_path: str) -> int:
    """Read the total rowcount of a parquet file from its metadata."""
    with _open_pq_file(pq_path) as pqf:
        return cast(int, pqf.metadata.num_rows)


@contextmanager
def _open_pq_file(pq_path: str, mode: str = "rb") -> Iterator[pq.ParquetFile]:
    filesystem = get_fs_for_path(pq_path)
    try:
        with filesystem.open(pq_path, mode=mode) as f, pq.ParquetFile(f) as pqf:
            yield pqf
    except Exception as e:
        raise RuntimeError(f"Unable to open {pq_path} as a parquet file") from e
