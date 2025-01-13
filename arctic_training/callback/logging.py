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

from typing import Any

from arctic_training.logging import logger

pre_init_log_cb = (
    "pre-init",
    lambda self: logger.info(f"Initializing {self.__class__.__name__}"),
)
post_init_log_cb = (
    "post-init",
    lambda self: logger.info(f"Initialized {self.__class__.__name__}"),
)


def _log_loss_value(self, loss: Any) -> Any:
    if self.global_step % self.config.loss_log_interval == 0:
        logger.info(f"Global Step: {self.global_step}/{self.training_horizon}, Loss: {loss}")
    return loss


post_loss_log_cb = ("post-loss", _log_loss_value)
