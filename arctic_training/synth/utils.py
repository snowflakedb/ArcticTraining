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


def import_error(**kwargs):
    raise ImportError("Please install the vllm package to use this class.")


def pass_function(**kwargs):
    pass


def recursive_to_dict(obj):
    # If the object has a __dict__ attribute, use it for conversion.
    if hasattr(obj, "__dict__"):
        return recursive_to_dict(obj.__dict__)
    # If it's a dictionary, process each key/value pair.
    elif isinstance(obj, dict):
        return {key: recursive_to_dict(value) for key, value in obj.items()}
    # If it's a list, process each element.
    elif isinstance(obj, list):
        return [recursive_to_dict(item) for item in obj]
    # If it's a tuple, process each element and return a tuple.
    elif isinstance(obj, tuple):
        return tuple(recursive_to_dict(item) for item in obj)
    # For other types (e.g., int, float, str), return as is.
    else:
        return obj
