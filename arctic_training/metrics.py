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

import functools
from deepspeed.utils.timer import SynchronizedWallClockTimer
from abc import ABC, abstractmethod

def estimate_flos(model, input_shape) -> int:
    pass

class Metric(ABC):
    def __init__(self, trainer):
        self.trainer = trainer
        self.setup(self.trainer)

    @abstractmethod
    def setup(self, trainer) -> None:
        ...

    @abstractmethod
    def get_metric(self) -> str:
        ...

class StepTimeMetric(Metric):
    def setup(self, trainer) -> None:
        self.timer = SynchronizedWallClockTimer.Timer("step")
        self.accumulated_times = []
        trainer.step = self._time_wrapper(trainer.step)

    def get_metric(self) -> str:
        mean_time = sum(self.accumulated_times) / len(self.accumulated_times)
        std_time = sum((t - mean_time) ** 2 for t in self.accumulated_times) / len(self.accumulated_times)
        self.accumulated_times = [] # reset times
        return f"Step time: {mean_time:.2f} ± {std_time:.2f} s"

    def _time_wrapper(self, method):
        """Wraps a method to measure execution time."""
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            self.timer.start()
            result = method(*args, **kwargs)
            self.timer.stop()
            elapsed_time = self.timer.elapsed() / 1000
            self.accumulated_times.append(elapsed_time)
            return result
        return wrapper

class IterTimeMetric(Metric):
    def setup(self, trainer) -> None:
        self.timer = SynchronizedWallClockTimer.Timer("iter")
        self.accumulated_times = []
        trainer.step = self._time_wrapper(trainer.train_dataloader.__iter__)

    def get_metric(self) -> str:
        mean_time = sum(self.accumulated_times) / len(self.accumulated_times)
        std_time = sum((t - mean_time) ** 2 for t in self.accumulated_times) / len(self.accumulated_times)
        self.accumulated_times = []
        return f"Iter time: {mean_time:.2f} ± {std_time:.2f} s"
    
    def _time_wrapper(self, method):
        """Wraps a method to measure execution time."""
        @functools.wraps(method)
        def wrapper(*args):
            iterator = method(*args)
            try:
                print("HERE", iterator)
                for item in iterator:
                    if self.timer.started_:
                        self.timer.stop()
                        elapsed_time = self.timer.elapsed() / 1000
                        self.accumulated_times.append(elapsed_time)
                    self.timer.start()
                    yield item
            finally:
                self.timer.stop()
                elapsed_time = self.timer.elapsed() / 1000
                self.accumulated_times.append(elapsed_time)

        return wrapper