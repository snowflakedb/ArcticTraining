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
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Union
from typing import cast

import torch
from deepspeed.utils.timer import SynchronizedWallClockTimer

from arctic_training.debug import get_mem_metrics
from arctic_training.utils import human_format_base10_number
from arctic_training.utils import human_format_secs

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


def gather_object(number: Union[float, int], world_size: int) -> List[Union[float, int]]:
    output = [None] * world_size
    torch.distributed.all_gather_object(output, number)
    return cast(List[Union[float, int]], output)


class Metrics:
    """Class for measuring, tracking, and reporting training metrics."""

    def __init__(self, trainer: "Trainer") -> None:
        self.enabled = trainer.config.train_log_iter_interval > 0
        if not self.enabled:
            return

        self.trainer = trainer
        self.summary_dict: Dict[str, Union[int, float]] = {}
        self.timers: Dict[str, SynchronizedWallClockTimer.Timer] = {}
        self.values: Dict[str, Union[int, float]] = defaultdict(float)

        # Store model size values for quickly calculating tflos later
        def numel_fn(p):
            return p.ds_numel if hasattr(p, "ds_tensor") else p.numel()

        model = self.trainer.model_unwrapped
        self.model_size = sum(numel_fn(p) for p in model.parameters())
        self.model_num_layers = model.config.num_hidden_layers
        self.model_hidden_size = model.config.hidden_size

        # Set max_iter based on when we expect to exit training
        if self.trainer.config.exit_iteration > 0:
            self.max_iter = min(self.trainer.config.exit_iteration, self.trainer.training_horizon)
        else:
            self.max_iter = self.trainer.training_horizon
        self.max_iter_pad = len(str(self.max_iter))

    def record(self, key: str, value: Union[int, float]) -> None:
        """Records a value in the metrics dictionary."""
        if not self.enabled:
            return
        if key in self.values:
            raise KeyError(
                f"Key {key} already exists. You are trying to write a value that has"
                " not yet been reported. This can happen if you try to write to a"
                " given value more than once in a training iteration loop."
            )
        self.values[key] = value

    def start_timer(self, key: str) -> None:
        """Starts a timer identified by `key`. If timer does not exist, one is created."""
        if not self.enabled:
            return
        if key not in self.timers:
            self.timers[key] = SynchronizedWallClockTimer().Timer(key)
        self.timers[key].start()

    def stop_timer(self, key: str) -> None:
        """Stops a timer identfied by `key` and records the elapsed time in seconds to the metrics dictionary."""
        if not self.enabled:
            return
        if key not in self.timers:
            raise KeyError(f"Timer {key} not started")
        self.timers[key].stop()
        self.values[f"{key}_time"] = self.timers[key].elapsed() / 1000

    def restart_timer(self, key: str) -> None:
        self.stop_timer(key)
        self.start_timer(key)

    def _estimate_decoder_transformer_tflos(self, seq_len: Union[int, float]) -> float:
        """Given a sequence length, estimates the number of floating point operations required to run the model."""
        return (
            seq_len * self.model_size * 2 * 4
            + self.model_num_layers * seq_len * seq_len * self.model_hidden_size * 2 * 2 * 4
        ) / 1e12

    def get_value(self, key: str) -> Union[int, float]:
        """Returns the value stored in the metrics dictionary for the given key."""
        return self.values[key]

    def print_summary(self) -> None:
        """Prints a summary of the metrics. If a value is not recorded by the Trainer, it is not included in the summary."""
        if not self.enabled:
            return

        self.summary_dict.clear()
        self.summary_dict["epoch"] = self.trainer.epoch_idx
        self.summary_dict["iter"] = self.trainer.global_step
        self.summary_dict["lr"] = self.trainer.model.lr_scheduler.get_last_lr()[0]

        tflos_total: float = 0.0
        if "seqlen" in self.values:
            tflos_total = sum(
                gather_object(
                    self._estimate_decoder_transformer_tflos(self.values["seqlen"]),
                    self.trainer.world_size,
                )
            )

        if "loss" in self.values:
            loss = sum(gather_object(self.values["loss"], self.trainer.world_size)) / self.trainer.world_size
            self.summary_dict["loss"] = loss

        if "iter_time" in self.values:
            iter_time_total = sum(gather_object(self.values["iter_time"], self.trainer.world_size))
            self.summary_dict["iter_time"] = iter_time_total / self.trainer.world_size
            if tflos_total > 0:
                self.summary_dict["iter_tflops"] = tflos_total / iter_time_total

        if "seqlen" in self.values:
            seq_len_total = sum(gather_object(self.values["seqlen"], self.trainer.world_size))
            self.summary_dict["seqlen"] = seq_len_total / self.trainer.world_size

        if "step_time" in self.values:
            step_time_total = sum(gather_object(self.values["step_time"], self.trainer.world_size))
            self.summary_dict["step_time"] = step_time_total / self.trainer.world_size
            if tflos_total > 0:
                self.summary_dict["step_tflops"] = tflos_total / step_time_total

        self.values.clear()

        summary_str = (
            "iter:"
            f" {self.summary_dict['iter']:>{self.max_iter_pad}}/{self.max_iter}"
            f" {100*self.summary_dict['iter']//self.max_iter:>3}%"
        )
        if "loss" in self.summary_dict:
            summary_str += f" | loss: {self.summary_dict['loss']:.4f}"
        if "iter_time" in self.summary_dict:
            summary_str += f" | iter time: {human_format_secs(self.summary_dict['iter_time'])}"
        if "iter_tflops" in self.summary_dict:
            summary_str += f" | iter tflops: {self.summary_dict['iter_tflops']:.1f}"
        summary_str += f" | lr: {self.summary_dict['lr']:.3E}"
        if "seqlen" in self.summary_dict:
            summary_str += f" | seqlen: {human_format_base10_number(self.summary_dict['seqlen'])}"
        if "step_time" in self.summary_dict:
            summary_str += f" | step time: {human_format_secs(self.summary_dict['step_time'])}"
        if "step_tflops" in self.summary_dict:
            summary_str += f" | step tflops: {self.summary_dict['step_tflops']:.1f}"
        summary_str += f" | epoch: {self.summary_dict['epoch']}"

        if self.trainer.global_rank == 0:
            # XXX: make configurable via yaml
            mem_metrics = get_mem_metrics()
            summary_str += f" | {mem_metrics}"

        if self.trainer.global_rank == 0:
            print(summary_str)
