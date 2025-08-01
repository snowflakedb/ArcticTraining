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

import sys

import pytest

from arctic_training.config.trainer import get_config
from arctic_training.config.trainer import load_user_module_from_path


def test_max_length_auto_setting():
    base_config = {
        "model": {"name_or_path": "distilgpt2"},
        "data": {"sources": ["HuggingFaceH4/ultrachat_200k"]},
    }

    # 1. When max_length is not provided, it should be set from the model config.
    config_without_max_length = get_config(base_config)
    assert config_without_max_length.data.max_length == 1024

    # 2. When max_length is provided, it should keep the provided value.
    config_with_max_length_dict = base_config.copy()
    config_with_max_length_dict["data"] = {"sources": ["HuggingFaceH4/ultrachat_200k"], "max_length": 512}
    config_with_max_length = get_config(config_with_max_length_dict)
    assert config_with_max_length.data.max_length == 512


def test_duplicate_key_fail(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
        data:
          max_length: 128
          max_length: 256
        """
    )

    with pytest.raises(ValueError, match=r"Duplicate .* key found"):
        get_config(config_file)


def test_load_user_module_with_relative_imports(tmp_path):
    """Test that load_user_module_from_path handles scripts with relative imports."""
    # Create a directory structure with multiple Python files
    user_code_dir = tmp_path / "user_code"
    user_code_dir.mkdir()

    # Create a helper module
    helper_file = user_code_dir / "helper.py"
    helper_file.write_text(
        """
def get_message():
    return "Hello from helper!"

HELPER_CONSTANT = 42
"""
    )

    # Create the main script that imports from helper
    main_script = user_code_dir / "main.py"
    main_script.write_text(
        """
from helper import get_message, HELPER_CONSTANT

def get_combined_message():
    return f"{get_message()} Value: {HELPER_CONSTANT}"

# This will be checked by the test
TEST_VALUE = get_combined_message()
"""
    )

    # Load the user module
    load_user_module_from_path(main_script)

    # Find the loaded module in sys.modules
    loaded_module = None
    for module_name, module in sys.modules.items():
        if "main" in module_name and hasattr(module, "TEST_VALUE"):
            loaded_module = module
            break

    # Verify the module was loaded correctly and relative imports worked
    assert loaded_module is not None, "Module was not loaded"
    assert hasattr(loaded_module, "TEST_VALUE"), "Module doesn't have expected attribute"
    assert loaded_module.TEST_VALUE == "Hello from helper! Value: 42", "Relative import failed"
