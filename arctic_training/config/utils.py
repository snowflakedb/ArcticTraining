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

import yaml


# From https://gist.github.com/pypt/94d747fe5180851196eb?permalink_comment_id=4015118#gistcomment-4015118
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(
                    f"Duplicate '{key}' key found in YAML on line"
                    f" {key_node.start_mark.line + 1}."
                )
            mapping.add(key)
        return super().construct_mapping(node, deep)
