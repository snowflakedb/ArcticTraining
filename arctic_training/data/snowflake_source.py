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

import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Optional

from pydantic import Field
from pydantic import model_validator
from typing_extensions import Self

from arctic_training.config.data import DataSourceConfig
from arctic_training.data.source import DataSource
from arctic_training.data.utils import DatasetType

if TYPE_CHECKING:
    from snowflake.snowpark import Session

_DATASET_URI_PATTERN = re.compile(r"^snow://dataset/([^/]+)/versions/([^/]+)$")


def _check_snowflake_ml_installed() -> None:
    """Check if snowflake-ml-python is installed."""
    try:
        import snowflake.ml  # noqa: F401
    except ImportError:
        raise ImportError(
            "snowflake-ml-python is required for Snowflake data sources. "
            "Install with: pip install 'arctic_training[snowflake]'"
        )


def get_default_snowflake_session() -> "Session":
    """
    Get or create a default Snowflake Session.

    This function attempts to get an active Snowpark session. If none exists,
    it creates a new session using default connection parameters.

    The session can be configured via:
    - Environment variables (SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD, etc.)
    - A Snowflake connection configuration file (~/.snowflake/connections.toml)
    - The SNOWFLAKE_DEFAULT_CONNECTION_NAME environment variable

    Returns:
        A Snowpark Session object.

    Raises:
        ImportError: If snowflake-snowpark-python is not installed.
        Exception: If session creation fails due to missing or invalid credentials.
    """
    _check_snowflake_ml_installed()

    from snowflake.snowpark import Session

    try:
        # Get an existing active session or create a new one using default connection
        # This will use environment variables or ~/.snowflake/connections.toml
        return Session.builder.getOrCreate()
    except Exception:
        from snowflake.ml._internal.utils.connection_params import SnowflakeLoginOptions

        # Fall back to SnowML's connection parameters
        config = SnowflakeLoginOptions()
        return Session.builder.configs(config).getOrCreate()  # noqa: F841


class SnowflakeSourceConfig(DataSourceConfig):
    """Configuration for Snowflake data sources.

    Supports three mutually exclusive modes:
    - sql: Execute a raw SQL query
    - table_name: Load all data from a table (generates SELECT * FROM table_name)
    - dataset_uri: Load from a versioned Snowflake Dataset
    """

    sql: Optional[str] = None
    """
    SQL query to execute against Snowflake.
    Example: 'SELECT col1, col2 FROM my_db.my_schema.my_table WHERE created_at > "2024-01-01"'
    """

    table_name: Optional[str] = None
    """
    Snowflake table reference in format [[db.]schema.]table_name.
    Examples: 'my_table', 'my_schema.my_table', 'my_db.my_schema.my_table'
    """

    dataset_uri: Optional[str] = None
    """
    Snowflake Dataset URI in format snow://dataset/<dataset_name>/versions/<version_name>.
    Where <dataset_name> is in format [[db.]schema.]dataset_name.
    Examples: 'snow://dataset/my_training_set/versions/v1', 'snow://dataset/my_schema.my_dataset/versions/v1'
    """

    column_mapping: Dict[str, str] = Field(default_factory=dict)
    """
    Optional mapping from source column names to target column names.
    If empty, data passes through unchanged.
    Example: {'source_col': 'target_col'} renames 'source_col' to 'target_col'.
    """

    limit: Optional[int] = None
    """Maximum number of rows to load. If None, loads all rows."""

    batch_size: int = 1024
    """Batch size for internal data retrieval."""

    @model_validator(mode="after")
    def validate_exactly_one_source(self) -> Self:
        """Ensure exactly one of sql, table_name, or dataset_uri is specified."""
        sources = [self.sql, self.table_name, self.dataset_uri]
        specified = sum(1 for s in sources if s is not None)
        if specified != 1:
            raise ValueError("Exactly one of 'sql', 'table_name', or 'dataset_uri' must be specified")

        # Auto-generate sql from table_name
        if self.table_name:
            self.sql = f"SELECT * FROM {self.table_name}"

        # Validate dataset_uri format if specified
        if self.dataset_uri:
            match = _DATASET_URI_PATTERN.match(self.dataset_uri)
            if not match:
                raise ValueError(
                    f"Invalid dataset_uri format: '{self.dataset_uri}'. "
                    "Expected format: 'snow://dataset/<dataset_name>/versions/<version_name>'"
                )

            # Validate the dataset_name component using Snowflake's identifier parser
            from snowflake.ml._internal.utils.identifier import parse_schema_level_object_identifier

            dataset_name = match.group(1)
            try:
                parse_schema_level_object_identifier(dataset_name)
            except ValueError as e:
                raise ValueError(f"Invalid dataset_name format in URI: {e}")

        return self


class SnowflakeDataSource(DataSource):
    """DataSource for loading data from Snowflake.

    Supports three modes:
    - SQL query: Execute arbitrary SQL and load results
    - Table: Load all data from a Snowflake table
    - Dataset: Load from a versioned Snowflake Dataset
    """

    name = "snowflake"
    config: SnowflakeSourceConfig

    session: Optional[Any] = None
    """
    Optional Snowpark Session to use for connecting to Snowflake.
    If None, a default session will be created using get_default_snowflake_session().
    """

    def load(self, config: SnowflakeSourceConfig, split: str) -> DatasetType:
        """Load data from Snowflake.

        Routes to the appropriate loading method based on config.
        """
        _check_snowflake_ml_installed()

        session = self.session or get_default_snowflake_session()

        if config.dataset_uri:
            return self._load_from_dataset(config, session=session)
        else:
            return self._load_from_sql(config, session=session)

    def _load_from_sql(self, config: SnowflakeSourceConfig, *, session: "Session") -> DatasetType:
        """Load data using a SQL query."""
        from snowflake.ml.data.data_connector import DataConnector

        # Create connector from SQL query
        connector = DataConnector.from_sql(config.sql, session=session)

        # Convert to HuggingFace dataset
        dataset = connector.to_huggingface_dataset(
            streaming=False,
            limit=config.limit,
            batch_size=config.batch_size,
        )

        return dataset

    def _load_from_dataset(self, config: SnowflakeSourceConfig, *, session: "Session") -> DatasetType:
        """Load data from a Snowflake Dataset."""
        from snowflake.ml.data.data_connector import DataConnector
        from snowflake.ml.dataset import load_dataset

        # Parse URI and load the Snowflake Dataset object
        assert config.dataset_uri is not None
        match = _DATASET_URI_PATTERN.match(config.dataset_uri)
        if not match:
            raise ValueError(f"Invalid dataset_uri format: '{config.dataset_uri}'")
        dataset_name, dataset_version = match.group(1), match.group(2)
        snow_dataset = load_dataset(session, dataset_name, dataset_version)

        # Create connector from the Dataset object
        connector = DataConnector.from_dataset(snow_dataset)

        # Convert to HuggingFace dataset
        dataset = connector.to_huggingface_dataset(
            streaming=False,
            limit=config.limit,
            batch_size=config.batch_size,
        )

        return dataset

    def post_load_callback(self, dataset: DatasetType) -> DatasetType:
        """Apply column mapping if provided."""
        if self.config.column_mapping:
            dataset = dataset.rename_columns(self.config.column_mapping)
        return dataset
