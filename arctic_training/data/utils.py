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

import hashlib
from functools import cache
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

import psutil
from datasets import Dataset
from datasets import IterableDataset
from torch.utils.data import DataLoader

DatasetType = Union[Dataset, IterableDataset]


@cache
def _get_node_fs_types() -> Dict[Path, str]:
    """Helper function to retrieve and cache filesystem types."""
    return {Path(r.mountpoint): r.fstype for r in psutil.disk_partitions(all=True)}


def _path_to_fs_type(path: Path) -> str:
    """
    Given a filesystem `path`, returns the filesystem type (ext, ext2, etc.).

    Note: Non-existent paths will return the filesystem type of `/`.
    """
    path = path.resolve()

    if path.is_symlink():
        path = path.readlink()  # py3.9+

    # Assuming at the end we percolate to `/` which is always there so the exit condition is assured
    try:
        return _get_node_fs_types()[path]
    except KeyError:
        return _path_to_fs_type(path.parent)


def is_local_fs(path: Path) -> bool:
    """Returns True if the `path` resides on a local filesystem, False otherwise."""
    local_node_fs_types = [
        "ext",
        "ext2",
        "ext3",
        "ext4",
        "reiserfs",
        "jfs",
        "xfs",
        "zfs",
        "xfs",
        "btrfs",
        "ntfs",
        "overlay",
    ]
    return _path_to_fs_type(path) in local_node_fs_types


def calculate_hash_from_args(*args: Any) -> str:
    hash_str = ""
    for arg in args:
        try:
            hash_str += str(arg)
        except Exception as e:
            raise ValueError(f"Failed to convert {arg} to string when calculating cache path. Error: {e}")
    return hashlib.md5(hash_str.encode()).hexdigest()


class OverfitOneBatchDataLoader(DataLoader):
    """A DataLoader that repeats the first batch of a base loader for testing purposes."""

    def __init__(self, base_loader: DataLoader, num_repeat: Optional[int] = None):
        self.size = len(base_loader) if num_repeat is None else num_repeat
        self.batch = next(iter(base_loader))

    def __len__(self):
        return self.size

    def __iter__(self):
        for _ in range(self.size):
            yield self.batch
