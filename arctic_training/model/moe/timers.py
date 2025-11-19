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

from deepspeed.utils.timer import SynchronizedWallClockTimer


class SynchronizedWallClockTimerSimple(SynchronizedWallClockTimer):
    """
    This is a simplified version of SynchronizedWallClockTimer that assumes that each timer does
    just start/stop and also takes care of not running the timers if its internal flag
    wall_clock_breakdown is False, so there is no need to litter the code with conditionals,
    leading to this:

        self.timers.start("test2")
        import time; time.sleep(0.5)
        self.timers.stop("test2")
        print(self.timers.times["test2"])

        # to activate the profiler
        timer = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
        # or if done at a later stage:
        timer = SynchronizedWallClockTimerSimple()
        .... some place later ...
        timer.wall_clock_breakdown = True

        the self.extra field allows for additional storage of

    """

    def __init__(self, wall_clock_breakdown=False):
        self.wall_clock_breakdown = wall_clock_breakdown
        super().__init__()  # creates self.timers
        self.times = defaultdict(float)

        # to allow additional token count stats
        self.token_counts = defaultdict(int)

    def start(self, name):
        """starts the clock if timing is enabled"""
        if not self.wall_clock_breakdown:
            return
        # pr0(f"{self(name)=}", force=True)
        self(name).start()

    def stop(self, name):
        """stops the clock and immediately stores the elapsed time"""
        # pr0(f"read: {self.wall_clock_breakdown} {name=} {self.times[name]=}")
        if not self.wall_clock_breakdown:
            self.times[name] = 0
            return
        self(name).stop()
        # if self.times[name] == 0:
        self.times[name] = self(name).elapsed(reset=False)
        # pr0(f"stored: {name=} {self.times[name]=}")

    def elapsed(self, name):
        """returns times stored by stop()"""
        return self.times[name]
        # if not self.wall_clock_breakdown:
        #     self.times[name] = 0
        # else:
        #     if self.times[name] == 0:
        #         # call only the first time, return cached value afterwards
        #         self.times[name] = super().elapsed(name, *args, **kwargs)
        # return self.times[name]

    def times(self):
        return self.times
