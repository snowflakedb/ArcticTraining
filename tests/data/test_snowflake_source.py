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

from arctic_training.data.snowflake_source import SnowflakeDatasetDataSource
from arctic_training.data.snowflake_source import SnowflakeDatasetSourceConfig
from arctic_training.data.snowflake_source import SnowflakeTableDataSource
from arctic_training.data.snowflake_source import SnowflakeTableSourceConfig


class TestSnowflakeTableSourceConfig:
    """Tests for SnowflakeTableSourceConfig validation."""

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
        config = SnowflakeTableSourceConfig(type="snowflake_table", table_name=table_name)
        assert config.table_name == table_name

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SnowflakeTableSourceConfig(type="snowflake_table", table_name="my_table")
        assert config.column_mapping == {}
        assert config.limit is None
        assert config.batch_size == 1024

    def test_custom_values(self):
        """Test that custom values are preserved."""
        config = SnowflakeTableSourceConfig(
            type="snowflake_table",
            table_name="my_table",
            column_mapping={"old_col": "new_col"},
            limit=1000,
            batch_size=512,
        )
        assert config.column_mapping == {"old_col": "new_col"}
        assert config.limit == 1000
        assert config.batch_size == 512


class TestSnowflakeDatasetSourceConfig:
    """Tests for SnowflakeDatasetSourceConfig validation."""

    @pytest.mark.parametrize(
        ("dataset_uri", "expected_dataset_uri", "expected_name", "expected_version"),
        [
            (
                "snow://dataset/my_dataset/versions/v1",
                "snow://dataset/my_dataset/versions/v1",
                "my_dataset",
                "v1",
            ),
            (
                'snow://dataset/"my-training_set"/versions/v1.0',
                'snow://dataset/"my-training_set"/versions/v1.0',
                '"my-training_set"',
                "v1.0",
            ),
            # Dataset names can also be qualified as [[db.]schema.]dataset_name
            (
                "snow://dataset/my_schema.my_dataset/versions/v1",
                "snow://dataset/my_schema.my_dataset/versions/v1",
                "my_schema.my_dataset",
                "v1",
            ),
            (
                "snow://dataset/my_db.my_schema.my_dataset/versions/v2",
                "snow://dataset/my_db.my_schema.my_dataset/versions/v2",
                "my_db.my_schema.my_dataset",
                "v2",
            ),
            # Quoted identifiers can be used to allow special characters (e.g. hyphens)
            (
                'snow://dataset/"my_db"."my_schema"."my-training_set"/versions/v1',
                'snow://dataset/"my_db"."my_schema"."my-training_set"/versions/v1',
                '"my_db"."my_schema"."my-training_set"',
                "v1",
            ),
        ],
    )
    def test_valid_dataset_uri(self, dataset_uri, expected_dataset_uri, expected_name, expected_version):
        """Test that valid dataset URIs are accepted and normalized."""
        config = SnowflakeDatasetSourceConfig(type="snowflake_dataset", dataset_uri=dataset_uri)
        assert config.dataset_uri == expected_dataset_uri
        name, version = config.parse_dataset_uri()
        assert name == expected_name
        assert version == expected_version

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
            SnowflakeDatasetSourceConfig(type="snowflake_dataset", dataset_uri=dataset_uri)
        assert "Invalid dataset_uri format" in str(exc_info.value) or "dataset_name format" in str(exc_info.value)

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SnowflakeDatasetSourceConfig(
            type="snowflake_dataset",
            dataset_uri="snow://dataset/my_dataset/versions/v1",
        )
        assert config.column_mapping == {}
        assert config.limit is None
        assert config.batch_size == 1024


class TestSnowflakeTableDataSource:
    """Tests for SnowflakeTableDataSource."""

    @patch("arctic_training.data.snowflake_source.get_default_snowflake_session")
    @patch("arctic_training.data.snowflake_source._check_snowflake_ml_installed")
    def test_load_calls_data_connector_with_sql(self, mock_check, mock_get_session):
        """Test that load() calls DataConnector.from_sql() with correct query."""
        # Setup mocks
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_dataset = Dataset.from_dict({"text": ["hello", "world"]})
        mock_connector_instance = MagicMock()
        mock_connector_instance.to_huggingface_dataset.return_value = mock_dataset
        mock_connector_cls = MagicMock()
        mock_connector_cls.from_sql.return_value = mock_connector_instance

        # Create config and data source
        config = SnowflakeTableSourceConfig(
            type="snowflake_table",
            table_name="my_db.my_schema.my_table",
            limit=100,
            batch_size=512,
        )

        mock_data_factory = MagicMock()
        data_source = SnowflakeTableDataSource(data_factory=mock_data_factory, config=config)

        # Patch the DataConnector import inside the load method
        with patch("arctic_training.data.snowflake_source.DataConnector", mock_connector_cls):
            result = data_source.load(config, split="train")

        # Verify SQL query was constructed correctly
        mock_connector_cls.from_sql.assert_called_once_with(
            "SELECT * FROM my_db.my_schema.my_table", session=mock_session
        )
        mock_connector_instance.to_huggingface_dataset.assert_called_once_with(
            streaming=False,
            limit=100,
            batch_size=512,
        )
        assert result == mock_dataset

    def test_post_load_callback_applies_column_mapping(self):
        """Test that post_load_callback applies column mapping."""
        mock_dataset = Dataset.from_dict({"old_col": ["a", "b"], "other": [1, 2]})
        mock_renamed_dataset = Dataset.from_dict({"new_col": ["a", "b"], "other": [1, 2]})
        mock_dataset.rename_columns = MagicMock(return_value=mock_renamed_dataset)

        config = SnowflakeTableSourceConfig(
            type="snowflake_table",
            table_name="my_table",
            column_mapping={"old_col": "new_col"},
        )

        mock_data_factory = MagicMock()
        data_source = SnowflakeTableDataSource(data_factory=mock_data_factory, config=config)

        result = data_source.post_load_callback(mock_dataset)

        mock_dataset.rename_columns.assert_called_once_with({"old_col": "new_col"})
        assert result == mock_renamed_dataset

    def test_post_load_callback_passthrough_without_mapping(self):
        """Test that post_load_callback passes through unchanged without mapping."""
        mock_dataset = Dataset.from_dict({"col": ["a", "b"]})
        mock_dataset.rename_columns = MagicMock()

        config = SnowflakeTableSourceConfig(
            type="snowflake_table",
            table_name="my_table",
            column_mapping={},  # Empty mapping
        )

        mock_data_factory = MagicMock()
        data_source = SnowflakeTableDataSource(data_factory=mock_data_factory, config=config)

        result = data_source.post_load_callback(mock_dataset)

        mock_dataset.rename_columns.assert_not_called()
        assert result == mock_dataset


class TestSnowflakeDatasetDataSource:
    """Tests for SnowflakeDatasetDataSource."""

    @pytest.mark.parametrize(
        ("dataset_uri", "expected_name", "expected_version"),
        [
            ("snow://dataset/my_dataset/versions/v1", "my_dataset", "v1"),
            ("snow://dataset/my_schema.my_dataset/versions/v1", "my_schema.my_dataset", "v1"),
            ("snow://dataset/my_db.my_schema.my_dataset/versions/v2", "my_db.my_schema.my_dataset", "v2"),
            (
                'snow://dataset/"my-training_set"/versions/v1.0',
                "my-training_set",
                "v1.0",
            ),
        ],
    )
    def test_load_calls_data_connector(self, dataset_uri, expected_name, expected_version):
        """Test that load() calls load_dataset and DataConnector correctly."""
        # Setup mocks
        mock_session = MagicMock()

        mock_snow_dataset = MagicMock()
        mock_load_dataset_fn = MagicMock(return_value=mock_snow_dataset)

        mock_hf_dataset = Dataset.from_dict({"messages": [["msg1"], ["msg2"]]})
        mock_connector_instance = MagicMock()
        mock_connector_instance.to_huggingface_dataset.return_value = mock_hf_dataset
        mock_connector_cls = MagicMock()
        mock_connector_cls.from_dataset.return_value = mock_connector_instance

        # Create config and data source
        config = SnowflakeDatasetSourceConfig(
            type="snowflake_dataset",
            dataset_uri=dataset_uri,
            limit=500,
            batch_size=256,
        )

        mock_data_factory = MagicMock()
        data_source = SnowflakeDatasetDataSource(data_factory=mock_data_factory, config=config)

        # Patch imports inside the load method
        with patch("arctic_training.data.snowflake_source._check_snowflake_ml_installed"), patch(
            "arctic_training.data.snowflake_source.get_default_snowflake_session", return_value=mock_session
        ), patch("arctic_training.data.snowflake_source.DataConnector", mock_connector_cls), patch(
            "arctic_training.data.snowflake_source.load_dataset", mock_load_dataset_fn
        ):
            result = data_source.load(config, split="train")

        # Verify load_dataset was called with parsed name and version
        mock_load_dataset_fn.assert_called_once_with(mock_session, expected_name, expected_version)
        # Verify DataConnector.from_dataset was called with the loaded dataset
        mock_connector_cls.from_dataset.assert_called_once_with(mock_snow_dataset)
        mock_connector_instance.to_huggingface_dataset.assert_called_once_with(
            streaming=False,
            limit=500,
            batch_size=256,
        )
        assert result == mock_hf_dataset

    def test_post_load_callback_applies_column_mapping(self):
        """Test that post_load_callback applies column mapping."""
        mock_dataset = Dataset.from_dict({"chat": ["a", "b"]})
        mock_renamed_dataset = Dataset.from_dict({"messages": ["a", "b"]})
        mock_dataset.rename_columns = MagicMock(return_value=mock_renamed_dataset)

        config = SnowflakeDatasetSourceConfig(
            type="snowflake_dataset",
            dataset_uri="snow://dataset/my_dataset/versions/v1",
            column_mapping={"chat": "messages"},
        )

        mock_data_factory = MagicMock()
        data_source = SnowflakeDatasetDataSource(data_factory=mock_data_factory, config=config)

        result = data_source.post_load_callback(mock_dataset)

        mock_dataset.rename_columns.assert_called_once_with({"chat": "messages"})
        assert result == mock_renamed_dataset


class TestImportError:
    """Tests for graceful handling when snowflake-ml-python is not installed."""

    def test_import_error_message(self):
        """Test that a helpful error message is raised when package is missing."""
        # Patch the import to simulate missing package
        with patch.dict("sys.modules", {"snowflake.ml": None}):
            from arctic_training.data.snowflake_source import _check_snowflake_ml_installed

            with pytest.raises(ImportError) as exc_info:
                _check_snowflake_ml_installed()

            assert "snowflake-ml-python is required" in str(exc_info.value)
            assert "pip install 'arctic_training[snowflake]'" in str(exc_info.value)
