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

import os
from contextlib import contextmanager
from pathlib import Path
import deepspeed.comm as dist
import math
import datetime

from contextlib import contextmanager


# FlopCounterMode leaks memory via ModuleTracker in `torch<2.6` - fixed in torch-2.6 - if using a lower version switch to a copy of a known good version of torch.utils.module_tracker.ModuleTracker
# BACKCOMPAT: can remove this workaround and the whole `arctic_training/back_compat/module_tracker.py` once we require `torch>=2.6` in dependencies
import torch
from packaging import version
from torch.utils.flop_counter import FlopCounterMode
if version.parse(torch.__version__) < version.parse("2.6"):
    import arctic_training.back_compat.torch.utils.module_tracker
    # override the leaky version with the non-leaky one copied from torch==2.6 - note this only monkey patches `torch.utils.flop_counter`
    torch.utils.flop_counter.ModuleTracker = arctic_training.back_compat.torch.utils.module_tracker.ModuleTracker

class StepFlopCounter:
    """
        This context manager counts floating point operations performed during the execution of the content of the context manager.

        It skips measuring till `start_iter` arrives (usually `start_iter=2` is a good setting - this is because the first iteration always takes more compute than the rest while things are being set up.

        To support variable compute iterations (e.g. when seqlen may vary) it includes an optional caching. As long as the `cache_key` is the same it'll run `FlopCounterMode` once and then will continue returning a cached value. If the `cache_key` hasn't been seen it'll run `FlopCounterMode` once per new key.

        note: flos rather than flops is used in variable names to designate "floating point operations" in order to prevent ambiguity with flops which are "floating point operations per second".

        note: currently there are no docs for `FlopCounterMode` so see notes here: https://github.com/pytorch/pytorch/issues/123800 - one important nuance - it ignores element-wise computations - so it reports less than real, but close enough for large models. Can't beat the convenience.

        Example:

        # don't count for the iteration 1
        step_flos_counter = StepFlopCounter(start_iter=2)
        for iter, batch in enumerate(self.train_batches):
            # cache based on the seqlen of the batch
            with step_flos_counter(iter+1, cache_key=len(batch["input_ids"][0])):
                self.step(batch)
            print(f"{step_flos_counter.get_total_tflos()=}")
    """
    def __init__(self, start_iter=2, display=False):
        self.target_iter = start_iter
        self.flos_counter = FlopCounterMode(display=display)
        self.flos = 0
        self.cache = {}

    @contextmanager
    def __call__(self, iter, cache_key=0):
        """
            Arguments:
                iter: current iteration id counting from 1
                cache_key: whatever value you want to cache on - e.g. seqlen

        """

#        if iter in [2,3,4,5,6]:
        if iter > 1:
            with self.flos_counter:
                yield
            self.flos = self.flos_counter.get_total_flops()
            return
        else:
            yield
            return

        # skip first steps while things are unstable
        if iter < self.target_iter:
            yield
            return

        # avoid recalculating and pull from cache instead
        if cache_key in self.cache:
            print(f"Cache hit for {iter=} {cache_key=}")
            self.flos = self.cache[cache_key]
            yield
            return

        with self.flos_counter:
            yield
        self.flos = self.flos_counter.get_total_flops()
        self.cache[cache_key] = self.flos

    def get_total_tflos(self):
        return self.flos / 1e12

import torch
def gather_sum_number(number, device, group=None):
    tensor = torch.tensor(number).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor.item()

def gather_number(number, device, group=None):
    tensor = torch.tensor(number).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MEAN, group=group)
    return tensor.item()

def gather_mean_number(number, device, group=None):
    tensor = torch.tensor(number).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MEAN, group=group)
    return tensor.item()

def gather_sum_tensor(tensor, device, group=None):
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    return tensor

def gather_mean_tensor(tensor, device, group=None):
    dist.all_reduce(t, op=dist.ReduceOp.MEAN, group=group)
    return tensor

def gather_object(number, device, group=None):
    """ returns a list of objects """
    # XXX: which world size? probably better to always specify the group explicitly and derive from it to avoid bugs and assumptions
    output = [None for _ in range(get_world_size())]
    torch.distributed.all_gather_object(output, number, group=group)
    return output




def human_format_base2_number(num: float, suffix: str = "") -> str:
    if num == 0:
        return f"0{suffix}"

    units = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"]
    exponent = min(int(math.log(abs(num), 1024)), len(units) - 1)
    value = num / (1024**exponent)

    return f"{value:_.1f}{units[exponent]}{suffix}"


def human_format_base10_number(num: float, suffix: str = "") -> str:
    if num == 0:
        return f"0{suffix}"

    units = ["", "K", "M", "B", "T", "Qa", "Qi"]  # Qa: Quadrillion, Qi: Quintillion
    exponent = min(int(math.log(abs(num), 1000)), len(units) - 1)
    value = num / (1000**exponent)

    return f"{value:_.2f}{units[exponent]}{suffix}"

def human_format_secs(secs):
    """
    - less than a minute format into seconds with decimals: "%s.%msec"
    - one minute and over use "%H:%M:%S" format
    - if over one day use: "X days, %H:%M:%S" format
    """
    if secs < 60:
        return f"{secs:.3f}s"
    else:
        return str(datetime.timedelta(seconds=secs)).split('.')[0]
