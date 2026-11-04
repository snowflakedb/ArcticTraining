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

"""Packed-sequence tiled rematerialization and native CPU activation offload for
ArcticTraining, delegating to DeepSpeed's Hugging Face integration
(``deepspeed.runtime.activation_checkpointing.hf_tiled_rematerialization``).

Needs a DeepSpeed build that ships that module (sfc-gh-truwase/DeepSpeed,
docs/cpu-checkpointing-perf-note branch or newer) ahead of the installed ``deepspeed`` on
PYTHONPATH. ArcticTraining's SFT collator produces only ``position_ids`` for packed rows;
the DeepSpeed integration derives ``cu_seqlens`` / ``seq_idx`` from them.
"""

from deepspeed.runtime.activation_checkpointing.hf_tiled_rematerialization import (  # noqa: F401
    _install_top_level_position_ids_capture,
)
from deepspeed.runtime.activation_checkpointing.hf_tiled_rematerialization import (
    enable_native_activation_checkpoint_offload,
)
from deepspeed.runtime.activation_checkpointing.hf_tiled_rematerialization import (
    install_packed_seq_tiled_rematerialize,
)


def install_packed_seq_tiled_remat(model, token_budget):
    """Patch the model's mixers for packed-sequence tiled rematerialization."""
    return install_packed_seq_tiled_rematerialize(model, token_budget)


__all__ = ["install_packed_seq_tiled_remat", "enable_native_activation_checkpoint_offload"]
