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
import sys
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Sequence
from typing import Type
from typing import TypeVar
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel
from torch import Tensor
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .many_parquet_dataset import ManyParquetDataset
from .typing_utils import NDArrayOfFloat
from .typing_utils import NDArrayOfUint64

T_ConfigFromFile = TypeVar("T_ConfigFromFile", bound=BaseModel)

logger = logging.getLogger(__name__)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool sequence of vectors into a single vector via averaging."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def first_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pools the hidden states by selecting the first non-padding token representation
    for each sequence."""
    batch_size = last_hidden_states.shape[0]
    row = torch.arange(batch_size, device=last_hidden_states.device)
    col = attention_mask.argmax(dim=1)  # position of the first non-padding token
    return last_hidden_states[row, col]


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pools the hidden states by selecting the last non-padding token representation
    for each sequence."""
    batch_size = last_hidden_states.shape[0]
    row = torch.arange(batch_size, device=last_hidden_states.device)
    col = attention_mask.sum(dim=1) - 1  # position of the last non-padding token
    return last_hidden_states[row, col]


def normalize_embeddings(embedings_matrix: NDArrayOfFloat, inplace: bool = False) -> None:
    assert embedings_matrix.ndim == 2
    norms = np.linalg.norm(embedings_matrix, axis=1, keepdims=True)
    out = embedings_matrix if inplace else None
    return np.divide(embedings_matrix, norms, out=out)


def np_2darray_from_pa_list_array(
    fixed_size_list_array: pa.ChunkedArray,
) -> NDArrayOfFloat:
    """Converts from a fixed sized list array in Arrow to a Numpy 2D array."""
    embed_dim = len(fixed_size_list_array[0])
    res = fixed_size_list_array.combine_chunks().flatten().to_numpy().reshape(-1, embed_dim)
    return cast(NDArrayOfFloat, res)


def read_embeddings_from_pq(
    pq_paths: Sequence[str | Path],
    id_coumn_name: str = "ID",
    vector_colunn_name: str = "VECTOR",
    vector_memmap_file: str | Path | None = None,
) -> tuple[NDArrayOfUint64, NDArrayOfFloat]:
    ds = ManyParquetDataset.from_filepaths([str(p) for p in pq_paths])
    if vector_memmap_file is None:
        table = ds.read_all_as_table(columns_subset=[id_coumn_name, vector_colunn_name])
        ids = table.column(id_coumn_name).to_numpy()
        vectors = np_2darray_from_pa_list_array(table.column(vector_colunn_name))
    else:
        # Determine output shape from parquet metadata.
        num_rows = ds.num_rows
        with pq.ParquetFile(pq_paths[0]) as pqf:
            arrow_schema = pqf.metadata.schema.to_arrow_schema()
        vector_dim = arrow_schema.field(vector_colunn_name).type.list_size

        # Initialize output arrays, memory-mapping the vectors.
        ids = np.zeros(num_rows, dtype=np.uint64)
        vectors = np.memmap(
            vector_memmap_file,
            dtype=np.float32,
            mode="w+",
            shape=(num_rows, vector_dim),
        )

        # Read the input in batches into the output objects.
        start_idx = 0
        with tqdm(total=num_rows, desc="Reading embeddings to memmap", unit="row") as pbar:
            for batch_table in ds.stream_tables(columns_subset=[id_coumn_name, vector_colunn_name]):
                end_idx = start_idx + len(batch_table)
                ids[start_idx:end_idx] = batch_table.column(id_coumn_name).to_numpy()
                vectors[start_idx:end_idx] = np_2darray_from_pa_list_array(batch_table.column(vector_colunn_name))
                start_idx = end_idx
                pbar.update(len(batch_table))
    return ids, vectors


def run_config_file_cli(
    main_fn: Callable[[T_ConfigFromFile], Any],
    config_cls: Type[T_ConfigFromFile],
    logging_format: str = "%(asctime)s %(name)s [%(levelname)s] %(message)s",
    logging_level: int = logging.INFO,
) -> None:
    """Runs a CLI interface for a main function that takes a single BaseModel
    subclass as its single argument.

    NOTE: Any return object from `main_fn` will be ignored.
    """
    # Parse the config file.
    if len(sys.argv) != 2:
        raise RuntimeError("You must provide exactly one argument: the path to the config file")
    config_file_path = Path(sys.argv[1])
    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file_path}")
    config = config_cls.model_validate_json(config_file_path.read_text())

    # Set up logging.
    logging.basicConfig(level=logging_level, format=logging_format)

    # Run with proper logging redirect around progress bars.
    with logging_redirect_tqdm():
        logger.info(f"Running CLI with config:\n{config}")
        main_fn(config)
