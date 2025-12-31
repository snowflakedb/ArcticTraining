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

"""
Set up script to populate a Snowflake account with training data from HuggingFace.

This script:
1. Downloads the stas/gutenberg-100 dataset from HuggingFace
2. Creates a Snowflake database and schema
3. Uploads the data as a Snowflake table
4. Creates a Snowflake Dataset from that table

Prerequisites:
- Install arctic_training with snowflake extras: pip install 'arctic_training[snowflake]'
- Configure Snowflake credentials via:
  - Environment variables (SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD)
  - Config file (~/.snowflake/connections.toml)
  - SNOWFLAKE_DEFAULT_CONNECTION_NAME environment variable

Usage:
    python setup_snowflake.py [--database DATABASE] [--schema SCHEMA] [--sample-count N]
"""

import argparse

import pandas as pd
from datasets import load_dataset
from snowflake.ml.dataset import create_from_dataframe
from snowflake.snowpark import Session


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up Snowflake with training data from HuggingFace")
    parser.add_argument(
        "--database",
        type=str,
        default="ARCTIC_TRAINING",
        help="Snowflake database name (default: ARCTIC_TRAINING)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="CAUSAL_DEMO",
        help="Snowflake schema name (default: CAUSAL_DEMO)",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="GUTENBERG_100",
        help="Snowflake table name (default: GUTENBERG_100)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="GUTENBERG_DATASET",
        help="Snowflake Dataset name (default: GUTENBERG_DATASET)",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="v1",
        help="Snowflake Dataset version (default: v1)",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="stas/gutenberg-100",
        help="HuggingFace dataset to download (default: stas/gutenberg-100)",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Number of samples to upload (default: 100)",
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing table and dataset if they exist",
    )
    return parser.parse_args()


def download_hf_dataset(dataset_name: str, sample_count: int) -> pd.DataFrame:
    """Download dataset from HuggingFace and convert to pandas DataFrame."""
    print(f"Downloading HuggingFace dataset: {dataset_name}")

    # Load the dataset
    hf_dataset = load_dataset(dataset_name, split=f"train[:{sample_count}]")

    # Convert to pandas DataFrame
    df = hf_dataset.to_pandas()

    # Ensure consistent column naming (uppercase for Snowflake)
    df.columns = [col.upper() for col in df.columns]

    print(f"Downloaded {len(df)} samples with columns: {list(df.columns)}")
    return df


def create_snowflake_resources(
    session: Session,
    database: str,
    schema: str,
    table_name: str,
    dataset_name: str,
    dataset_version: str,
    df: pd.DataFrame,
    drop_existing: bool = False,
) -> None:
    """Create Snowflake database, schema, table, and dataset."""

    # Create database and schema
    print(f"Creating database: {database}")
    session.sql(f"CREATE DATABASE IF NOT EXISTS {database}").collect()

    print(f"Creating schema: {database}.{schema}")
    session.sql(f"CREATE SCHEMA IF NOT EXISTS {database}.{schema}").collect()

    # Set the context
    session.use_database(database)
    session.use_schema(schema)

    full_table_name = f"{database}.{schema}.{table_name}"
    full_dataset_name = f"{database}.{schema}.{dataset_name}"

    if drop_existing:
        print(f"Dropping existing table (if exists): {full_table_name}")
        session.sql(f"DROP TABLE IF EXISTS {full_table_name}").collect()

        # Note: Snowflake Datasets cannot be dropped via SQL, they need to be
        # deleted via the Dataset API or will be overwritten

    # Create table from pandas DataFrame
    print(f"Creating table: {full_table_name}")
    snowpark_df = session.create_dataframe(df)
    snowpark_df.write.mode("overwrite").save_as_table(full_table_name)

    # Verify table creation
    row_count = session.sql(f"SELECT COUNT(*) FROM {full_table_name}").collect()[0][0]
    print(f"Table created with {row_count} rows")

    # Create Snowflake Dataset from the table
    print(f"Creating Snowflake Dataset: {full_dataset_name} (version: {dataset_version})")

    # Read the table as a Snowpark DataFrame for dataset creation
    table_df = session.table(full_table_name)

    # Create the dataset
    snow_dataset = create_from_dataframe(
        session=session,
        name=full_dataset_name,
        version=dataset_version,
        input_dataframe=table_df,
    )

    print(f"Dataset created: {snow_dataset.fully_qualified_name}")
    print(f"Dataset URI: snow://dataset/{full_dataset_name}/versions/{dataset_version}")


def main() -> None:
    args = get_args()

    print("=" * 60)
    print("Snowflake Set Up Script for Arctic Training")
    print("=" * 60)

    # Download HuggingFace dataset
    df = download_hf_dataset(args.hf_dataset, args.sample_count)

    # Connect to Snowflake
    print("\nConnecting to Snowflake...")
    session = Session.builder.getOrCreate()
    print(f"Connected to account: {session.get_current_account()}")

    try:
        # Create Snowflake resources
        create_snowflake_resources(
            session=session,
            database=args.database,
            schema=args.schema,
            table_name=args.table_name,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            df=df,
            drop_existing=args.drop_existing,
        )

        print("\n" + "=" * 60)
        print("Set up completed successfully!")
        print("=" * 60)
        print("\nCreated resources:")
        print(f"  - Table: {args.database}.{args.schema}.{args.table_name}")
        print(
            "  - Dataset:"
            f" snow://dataset/{args.database}.{args.schema}.{args.dataset_name}/versions/{args.dataset_version}"
        )
        print("\nYou can now run the training configs:")
        print("  - run-causal-snowflake-sql.yml")
        print("  - run-causal-snowflake-table.yml")
        print("  - run-causal-snowflake-dataset.yml")

    finally:
        session.close()


if __name__ == "__main__":
    main()
