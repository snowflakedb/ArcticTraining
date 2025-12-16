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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from datasets import Dataset
from pydantic import ValidationError
from snowflake.ml.data.data_connector import DataConnector

from arctic_training.data.snowflake_source import SnowflakeDataSource
from arctic_training.data.snowflake_source import SnowflakeSourceConfig


class TestSnowflakeSourceConfig:
    """Tests for SnowflakeSourceConfig validation."""

    # ===== SQL Mode Tests =====

    def test_valid_sql_query(self):
        """Test that valid SQL queries are accepted."""
        sql = "SELECT col1, col2 FROM my_db.my_schema.my_table WHERE id > 100"
        config = SnowflakeSourceConfig(type="snowflake", sql=sql)
        assert config.sql == sql

    def test_sql_with_custom_options(self):
        """Test that custom options are preserved with sql mode."""
        config = SnowflakeSourceConfig(
            type="snowflake",
            sql="SELECT * FROM my_table",
            column_mapping={"old_col": "new_col"},
            limit=1000,
            batch_size=512,
        )
        assert config.sql == "SELECT * FROM my_table"
        assert config.column_mapping == {"old_col": "new_col"}
        assert config.limit == 1000
        assert config.batch_size == 512

    # ===== Table Name Mode Tests =====

    @pytest.mark.parametrize(
        "table_name",
        [
            "my_table",
            "my_schema.my_table",
            "my_db.my_schema.my_table",
        ],
    )
    def test_valid_table_name(self, table_name: str):
        """Test that [[db.]schema.]table format is accepted."""
        config = SnowflakeSourceConfig(type="snowflake", table_name=table_name)
        assert config.table_name == table_name

    @pytest.mark.parametrize(
        ("table_name", "expected_sql"),
        [
            ("my_table", "SELECT * FROM my_table"),
            ("my_schema.my_table", "SELECT * FROM my_schema.my_table"),
            ("my_db.my_schema.my_table", "SELECT * FROM my_db.my_schema.my_table"),
        ],
    )
    def test_sql_generated_from_table_name(self, table_name: str, expected_sql: str):
        """Test that sql field is auto-populated from table_name."""
        config = SnowflakeSourceConfig(type="snowflake", table_name=table_name)
        assert config.sql == expected_sql

    def test_table_name_with_custom_options(self):
        """Test that custom options are preserved with table_name mode."""
        config = SnowflakeSourceConfig(
            type="snowflake",
            table_name="my_table",
            column_mapping={"old_col": "new_col"},
            limit=1000,
            batch_size=512,
        )
        assert config.column_mapping == {"old_col": "new_col"}
        assert config.limit == 1000
        assert config.batch_size == 512

    # ===== Dataset URI Mode Tests =====

    @pytest.mark.parametrize(
        "dataset_uri",
        [
            "snow://dataset/my_dataset/versions/v1",
            'snow://dataset/"my-training_set"/versions/v1.0',
            # Dataset names can also be qualified as [[db.]schema.]dataset_name
            "snow://dataset/my_schema.my_dataset/versions/v1",
            "snow://dataset/my_db.my_schema.my_dataset/versions/v2",
            # Quoted identifiers can be used to allow special characters (e.g. hyphens)
            'snow://dataset/"my_db"."my_schema"."my-training_set"/versions/v1',
        ],
    )
    def test_valid_dataset_uri(self, dataset_uri):
        """Test that valid dataset URIs are accepted."""
        config = SnowflakeSourceConfig(type="snowflake", dataset_uri=dataset_uri)
        assert config.dataset_uri == dataset_uri

    @pytest.mark.parametrize(
        "dataset_uri",
        [
            # Missing snow:// prefix
            "dataset/my_dataset/versions/v1",
            # Wrong base path
            "snow://my_dataset/v1",
            # Missing version segment
            "snow://dataset/my_dataset",
            # Too many qualifiers: db.schema.dataset.extra
            "snow://dataset/a.b.c.d/versions/v1",
            # Empty identifier segments
            "snow://dataset/.a/versions/v1",
            "snow://dataset/a../versions/v1",
        ],
    )
    def test_invalid_dataset_uri(self, dataset_uri: str):
        """Test that invalid dataset URIs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SnowflakeSourceConfig(type="snowflake", dataset_uri=dataset_uri)
        assert "Invalid dataset_uri format" in str(exc_info.value) or "dataset_name format" in str(exc_info.value)

    def test_dataset_uri_with_custom_options(self):
        """Test that custom options are preserved with dataset_uri mode."""
        config = SnowflakeSourceConfig(
            type="snowflake",
            dataset_uri="snow://dataset/my_dataset/versions/v1",
            column_mapping={"chat": "messages"},
            limit=500,
            batch_size=256,
        )
        assert config.column_mapping == {"chat": "messages"}
        assert config.limit == 500
        assert config.batch_size == 256

    # ===== Default Values Tests =====

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SnowflakeSourceConfig(type="snowflake", sql="SELECT * FROM my_table")
        assert config.column_mapping == {}
        assert config.limit is None
        assert config.batch_size == 1024

    # ===== One-of Validation Tests =====

    def test_error_when_no_source_specified(self):
        """Test that error is raised when no source is specified."""
        with pytest.raises(ValidationError) as exc_info:
            SnowflakeSourceConfig(type="snowflake")
        assert "Exactly one of 'sql', 'table_name', or 'dataset_uri' must be specified" in str(exc_info.value)

    def test_error_when_multiple_sources_specified(self):
        """Test that error is raised when multiple sources are specified."""
        # sql + table_name
        with pytest.raises(ValidationError) as exc_info:
            SnowflakeSourceConfig(type="snowflake", sql="SELECT * FROM t", table_name="my_table")
        assert "Exactly one of 'sql', 'table_name', or 'dataset_uri' must be specified" in str(exc_info.value)

        # sql + dataset_uri
        with pytest.raises(ValidationError) as exc_info:
            SnowflakeSourceConfig(
                type="snowflake", sql="SELECT * FROM t", dataset_uri="snow://dataset/my_dataset/versions/v1"
            )
        assert "Exactly one of 'sql', 'table_name', or 'dataset_uri' must be specified" in str(exc_info.value)

        # table_name + dataset_uri
        with pytest.raises(ValidationError) as exc_info:
            SnowflakeSourceConfig(
                type="snowflake", table_name="my_table", dataset_uri="snow://dataset/my_dataset/versions/v1"
            )
        assert "Exactly one of 'sql', 'table_name', or 'dataset_uri' must be specified" in str(exc_info.value)

        # all three
        with pytest.raises(ValidationError) as exc_info:
            SnowflakeSourceConfig(
                type="snowflake",
                sql="SELECT * FROM t",
                table_name="my_table",
                dataset_uri="snow://dataset/my_dataset/versions/v1",
            )
        assert "Exactly one of 'sql', 'table_name', or 'dataset_uri' must be specified" in str(exc_info.value)


class TestSnowflakeDataSource:
    """Tests for SnowflakeDataSource."""

    # ===== SQL Mode Tests =====

    @patch.object(DataConnector, "from_sql")
    def test_load_with_sql(self, mock_from_sql):
        """Test that load() calls DataConnector.from_sql() with the provided SQL."""
        # Setup mocks
        mock_session = MagicMock()

        mock_dataset = Dataset.from_dict({"col1": ["a", "b"], "col2": [1, 2]})
        mock_connector_instance = MagicMock()
        mock_connector_instance.to_huggingface_dataset.return_value = mock_dataset
        mock_from_sql.return_value = mock_connector_instance

        # Create config and data source
        sql = "SELECT col1, col2 FROM my_db.my_schema.my_table WHERE id > 100"
        config = SnowflakeSourceConfig(
            type="snowflake",
            sql=sql,
            limit=100,
            batch_size=512,
        )

        mock_data_factory = MagicMock()
        data_source = SnowflakeDataSource(data_factory=mock_data_factory, config=config)
        data_source.session = mock_session

        result = data_source.load(config, split="train")

        # Verify SQL query was passed correctly
        mock_from_sql.assert_called_once_with(sql, session=mock_session)
        mock_connector_instance.to_huggingface_dataset.assert_called_once_with(
            streaming=False,
            limit=100,
            batch_size=512,
        )
        assert result == mock_dataset

    # ===== Table Name Mode Tests =====

    @patch.object(DataConnector, "from_sql")
    def test_load_with_table_name(self, mock_from_sql):
        """Test that load() with table_name generates correct SQL."""
        # Setup mocks
        mock_session = MagicMock()

        mock_dataset = Dataset.from_dict({"text": ["hello", "world"]})
        mock_connector_instance = MagicMock()
        mock_connector_instance.to_huggingface_dataset.return_value = mock_dataset
        mock_from_sql.return_value = mock_connector_instance

        # Create config and data source
        config = SnowflakeSourceConfig(
            type="snowflake",
            table_name="my_db.my_schema.my_table",
            limit=100,
            batch_size=512,
        )

        mock_data_factory = MagicMock()
        data_source = SnowflakeDataSource(data_factory=mock_data_factory, config=config)
        data_source.session = mock_session

        result = data_source.load(config, split="train")

        # Verify SQL query was constructed correctly from table_name
        mock_from_sql.assert_called_once_with("SELECT * FROM my_db.my_schema.my_table", session=mock_session)
        mock_connector_instance.to_huggingface_dataset.assert_called_once_with(
            streaming=False,
            limit=100,
            batch_size=512,
        )
        assert result == mock_dataset

    # ===== Dataset URI Mode Tests =====

    @pytest.mark.parametrize(
        ("dataset_uri", "expected_name", "expected_version"),
        [
            ("snow://dataset/my_dataset/versions/v1", "my_dataset", "v1"),
            # Dataset names can also be qualified as [[db.]schema.]dataset_name
            ("snow://dataset/my_schema.my_dataset/versions/v1", "my_schema.my_dataset", "v1"),
            ("snow://dataset/my_db.my_schema.my_dataset/versions/v2", "my_db.my_schema.my_dataset", "v2"),
            # Quoted identifiers can be used to allow special characters (e.g. hyphens)
            ('snow://dataset/"my-training_set"/versions/v1.0', '"my-training_set"', "v1.0"),
            (
                'snow://dataset/"my_db"."my_schema"."my-training_set"/versions/v1',
                '"my_db"."my_schema"."my-training_set"',
                "v1",
            ),
        ],
    )
    @patch("snowflake.ml.dataset.load_dataset")
    @patch.object(DataConnector, "from_dataset")
    def test_load_with_dataset_uri(
        self, mock_from_dataset, mock_load_dataset, dataset_uri, expected_name, expected_version
    ):
        """Test that load() with dataset_uri calls DataConnector.from_dataset() correctly."""
        # Setup mocks
        mock_session = MagicMock()

        mock_snow_dataset = MagicMock()
        mock_load_dataset.return_value = mock_snow_dataset

        mock_hf_dataset = Dataset.from_dict({"messages": [["msg1"], ["msg2"]]})
        mock_connector_instance = MagicMock()
        mock_connector_instance.to_huggingface_dataset.return_value = mock_hf_dataset
        mock_from_dataset.return_value = mock_connector_instance

        # Create config and data source
        config = SnowflakeSourceConfig(
            type="snowflake",
            dataset_uri=dataset_uri,
            limit=500,
            batch_size=256,
        )

        mock_data_factory = MagicMock()
        data_source = SnowflakeDataSource(data_factory=mock_data_factory, config=config)
        data_source.session = mock_session

        result = data_source.load(config, split="train")

        # Verify load_dataset was called with correct arguments
        mock_load_dataset.assert_called_once_with(mock_session, expected_name, expected_version)
        # Verify DataConnector.from_dataset was called with the loaded dataset
        mock_from_dataset.assert_called_once_with(mock_snow_dataset)
        mock_connector_instance.to_huggingface_dataset.assert_called_once_with(
            streaming=False,
            limit=500,
            batch_size=256,
        )
        assert result == mock_hf_dataset

    # ===== Post Load Callback Tests =====

    def test_post_load_callback_applies_column_mapping(self):
        """Test that post_load_callback applies column mapping."""
        mock_dataset = Dataset.from_dict({"old_col": ["a", "b"], "other": [1, 2]})
        mock_renamed_dataset = Dataset.from_dict({"new_col": ["a", "b"], "other": [1, 2]})
        mock_dataset.rename_columns = MagicMock(return_value=mock_renamed_dataset)

        config = SnowflakeSourceConfig(
            type="snowflake",
            sql="SELECT * FROM my_table",
            column_mapping={"old_col": "new_col"},
        )

        mock_data_factory = MagicMock()
        data_source = SnowflakeDataSource(data_factory=mock_data_factory, config=config)

        result = data_source.post_load_callback(mock_dataset)

        mock_dataset.rename_columns.assert_called_once_with({"old_col": "new_col"})
        assert result == mock_renamed_dataset

    def test_post_load_callback_passthrough_without_mapping(self):
        """Test that post_load_callback passes through unchanged without mapping."""
        mock_dataset = Dataset.from_dict({"col": ["a", "b"]})
        mock_dataset.rename_columns = MagicMock()

        config = SnowflakeSourceConfig(
            type="snowflake",
            sql="SELECT * FROM my_table",
            column_mapping={},  # Empty mapping
        )

        mock_data_factory = MagicMock()
        data_source = SnowflakeDataSource(data_factory=mock_data_factory, config=config)

        result = data_source.post_load_callback(mock_dataset)

        mock_dataset.rename_columns.assert_not_called()
        assert result == mock_dataset
