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


def validate_callback(self):
    if self.model.is_gradient_accumulation_boundary():
        if self.config.eval_frequency > 0 and (self.model.global_steps + 1) % self.config.eval_frequency == 0:
            self.validate()


def earlystop_callback(self):
    if self.model.is_gradient_accumulation_boundary():
        if self.model.global_steps >= self.get_training_horizon():
            self.early_stop = True
