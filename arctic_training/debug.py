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


import gc

import torch.distributed as dist
from deepspeed.accelerator import get_accelerator

can_run_pynvml = True
try:
    import pynvml
except Exception:
    can_run_pynvml = False


pynvml_handle = None


def get_mem_metrics():
    global pynvml_handle

    if not can_run_pynvml:
        return ""

    gc.collect()

    if pynvml_handle is None:
        pynvml.nvmlInit()
        rank = dist.get_rank() if dist.is_initialized() else 0
        pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
        # pynvml.nvmlShutdown()
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(pynvml_handle)
    nv_mem = memory_info.used

    summary = " | ".join(
        [
            f"MA {round(get_accelerator().memory_allocated() / 2**30, 2):0.2f} GB",
            f"Max_MA {round(get_accelerator().max_memory_allocated() / 2**30, 2):0.2f} GB",
            f"NV {round(nv_mem / 2**30, 2):0.2f} GB",
        ]
    )

    # get the peak memory to report correct data, so reset the counter for the next call
    # this will lead to wrong peak reports if `see_mem_usage` is also used during the run,
    # as it resets the peak counter and there is only one counter
    get_accelerator().reset_peak_memory_stats()

    return summary
