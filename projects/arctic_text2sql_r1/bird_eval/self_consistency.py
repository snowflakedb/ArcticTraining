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

from typing import Any
from typing import Hashable

import numpy as np
import pandas as pd


def efficient_soft_df_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """
    Computes the "soft denotation" similarity between two DataFrames:
      - For each column, get the frequency counts of each distinct value in df1 and df2
      - Align them by the union of distinct values
      - Accumulate 'real agreement' vs. 'possible agreement'
      - Return total_real_agreement / total_possible_agreement

    Args:
        df1, df2 (pd.DataFrame): DataFrames to compare

    Returns:
        float: Similarity score in [0, 1].
    """
    # If either DataFrame is empty in rows or columns, similarity = 0
    if df1 is None or df2 is None or len(df1) == 0 or len(df2) == 0:
        return 0.0

    df1 = df1.copy()  # Precaution
    df2 = df2.copy()

    def _ensure_hashable_values(value: Any) -> Hashable:
        if isinstance(value, Hashable):
            return value
        return repr(value)

    # Use applymap for DataFrames
    df1 = df1.applymap(_ensure_hashable_values)
    df2 = df2.applymap(_ensure_hashable_values)

    # For rare cases where df has two columns with the same name (sic!)
    def _select_1d(df, col):
        c = df[col]
        if len(c.shape) == 2:
            return pd.DataFrame(c.stack().values)
        return c

    # Precompute value_counts for each column
    df1_counts = {col: _select_1d(df1, col).value_counts(dropna=False) for col in df1.columns}
    df2_counts = {col: _select_1d(df2, col).value_counts(dropna=False) for col in df2.columns}

    total_real_agreement = 0.0
    total_possible_agreement = 0.0

    # Union of all columns
    all_columns = df1.columns.union(df2.columns)

    for col in all_columns:
        vc1 = df1_counts.get(col)
        vc2 = df2_counts.get(col)

        if vc1 is None or vc1.empty:
            total_possible_agreement += vc2.to_numpy().sum()
            continue

        if vc2 is None or vc2.empty:
            total_possible_agreement += vc1.to_numpy().sum()
            continue

        # 1) Get union of distinct values in that column
        union_idx = pd.Index(pd.concat([vc1.index.to_frame(), vc2.index.to_frame()], axis=0).iloc[:, 0].unique())
        if union_idx.dtype != "object":
            union_idx = union_idx.astype(object)

        # 2) Reindex both frequency series to that union, fill missing with 0
        freq1 = vc1.reindex(union_idx, fill_value=0).values
        freq2 = vc2.reindex(union_idx, fill_value=0).values

        if np.nan in union_idx:
            freq1[union_idx.isnull()] += freq2[union_idx.isnull()]
            freq2[union_idx.isnull()] = 0

        # 3) Vectorized computations (avoiding DataFrame overhead)
        possible_agreement = np.maximum(freq1, freq2).sum()
        accumulated_difference = np.abs(freq1 - freq2).sum()
        real_agreement = possible_agreement - accumulated_difference

        # Accumulate column-wise
        total_real_agreement += real_agreement
        total_possible_agreement += possible_agreement

    # Avoid division by zero if possible_agreement == 0
    if total_possible_agreement == 0:
        return 0.0

    return total_real_agreement / total_possible_agreement


def calculate_similarity_matrix(
    candidate_sqls,
) -> np.ndarray:
    sql_len = len(candidate_sqls)
    similarity_matrix = np.zeros((sql_len, sql_len))
    for idx1 in range(sql_len):
        df1 = candidate_sqls[idx1]
        if df1 is not None:
            similarity_matrix[idx1, idx1] += 1
        for idx2 in range(idx1 + 1, sql_len):
            df2 = candidate_sqls[idx2]
            similarity = efficient_soft_df_similarity(df1, df2)
            similarity_matrix[idx1, idx2] += similarity
            similarity_matrix[idx2, idx1] += similarity
    return similarity_matrix
