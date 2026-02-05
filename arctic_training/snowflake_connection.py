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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snowflake.snowpark import Session


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
