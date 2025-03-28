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

from collections import defaultdict
from functools import wraps
from typing import List

import torch
from deepspeed.utils.timer import SynchronizedWallClockTimer


def gather_object(number, world_size, group=None) -> List:
    """returns a list of objects"""
    output = [None] * world_size
    torch.distributed.all_gather_object(output, number, group=group)
    return output


def disabled_early_exit_wrapper(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.enabled:
            return
        return method(self, *args, **kwargs)

    return wrapper


class Metrics:
    def __init__(self, trainer):
        self.trainer = trainer
        enabled = self.trainer.config.train_log_iter_interval > 0
        if not enabled:
            return

        self.timers = {
            key: SynchronizedWallClockTimer.Timer(key) for key in ["step", "iter"]
        }
        self.timers = defaultdict(lambda key: SynchronizedWallClockTimer.Timer(key))
        self.values = defaultdict(float)

        model = self.trainer.model_unwrapped

        def numel_fn(p):
            return p.ds_numel if hasattr(p, "ds_tensor") else p.numel()

        self.model_size = sum(numel_fn(p) for p in model.parameters())
        self.model_num_layers = model.config.num_hidden_layers
        self.model_hidden_size = model.config.hidden_size

    @disabled_early_exit_wrapper
    def record(self, key: str, value: float) -> None:
        self.values[key] = value

    @disabled_early_exit_wrapper
    def start(self, key: str) -> None:
        self.timers[key].start()

    @disabled_early_exit_wrapper
    def stop(self, key: str) -> None:
        self.timers[key].stop()
        self.values[key] = self.timers[key].elapsed() / 1000

    def _estimate_tflos(self, seq_len) -> float:
        return (
            seq_len * self.model_size * 2 * 4
            + self.model_num_layers
            * seq_len
            * seq_len
            * self.model_hidden_size
            * 2
            * 2
            * 4
        ) / 1e12

    @disabled_early_exit_wrapper
    def get_values(self) -> list[float]:
        return self.values

    @disabled_early_exit_wrapper
    def print_summary(self) -> None:
        summary_str = f"iter: {self.trainer.train_batch_idx}"

        if "loss" in self.values:
            loss = (
                sum(gather_object(self.values["loss"], self.trainer.world_size))
                / self.trainer.world_size
            )
            summary_str += f" | loss: {loss:.4f}"

        lr = self.trainer.model.lr_scheduler.get_last_lr()[0]
        summary_str += f" | lr: {lr:.4E}"

        tflos_total = 0
        if "seqlen" in self.values:
            seq_len_total = sum(
                gather_object(self.values["seqlen"], self.trainer.world_size)
            )
            tflos_total = sum(
                gather_object(
                    self._estimate_tflos(self.values["seqlen"]), self.trainer.world_size
                )
            )
            summary_str += f" | seqlen total: {seq_len_total:d}"

        if "iter" in self.values:
            iter_time_total = sum(
                gather_object(self.values["iter"], self.trainer.world_size)
            )
            summary_str += (
                f" | iter time: {iter_time_total/self.trainer.world_size:.4f}"
            )
            if tflos_total > 0:
                summary_str += f" | iter tflops: {tflos_total / iter_time_total:.1f}"

        if "step" in self.values:
            step_time_total = sum(
                gather_object(self.values["step"], self.trainer.world_size)
            )
            summary_str += (
                f" | step time: {step_time_total/self.trainer.world_size:.4f}"
            )
            if tflos_total > 0:
                summary_str += f" | step tflops: {tflos_total / step_time_total:.1f}"

        print(summary_str)
