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
from typing import Dict
from typing import List

import torch
from deepspeed.utils.timer import SynchronizedWallClockTimer
from tqdm import tqdm


def gather_object(number, world_size, group=None) -> List:
    """returns a list of objects"""
    output = [None] * world_size
    torch.distributed.all_gather_object(output, number, group=group)
    return output


class Metrics:
    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self.enabled = self.trainer.config.train_log_iter_interval > 0
        if not self.enabled:
            return

        self.timers: Dict[str, SynchronizedWallClockTimer.Timer] = {}
        self.values: Dict[str, float] = defaultdict(float)

        model = self.trainer.model_unwrapped

        def numel_fn(p):
            return p.ds_numel if hasattr(p, "ds_tensor") else p.numel()

        self.model_size = sum(numel_fn(p) for p in model.parameters())
        self.model_num_layers = model.config.num_hidden_layers
        self.model_hidden_size = model.config.hidden_size

        if self.trainer.config.exit_iteration > 0:
            self.max_iter = min(
                self.trainer.config.exit_iteration, self.trainer.training_horizon
            )
        else:
            self.max_iter = self.trainer.training_horizon

    def record(self, key: str, value: float) -> None:
        if not self.enabled:
            return
        self.values[key] = value

    def start(self, key: str) -> None:
        if not self.enabled:
            return
        if key not in self.timers:
            self.timers[key] = SynchronizedWallClockTimer().Timer(key)
        self.timers[key].start()

    def stop(self, key: str) -> None:
        if not self.enabled:
            return
        if key not in self.timers:
            raise KeyError(f"Timer {key} not started")
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

    def get_values(self, key: str) -> float:
        return self.values[key]

    def print_summary(self) -> None:
        if not self.enabled:
            return

        len_max_iter = len(str(self.max_iter))
        summary_str = (
            f"iter: {self.trainer.train_batch_idx:>{len_max_iter}} / {self.max_iter}"
        )

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

        tqdm.write(summary_str)
