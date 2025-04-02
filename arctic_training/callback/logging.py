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


def _pre_init_log(self) -> None:
    logger.info(f"Initializing {self.__class__.__name__}")


pre_init_log_cb = (
    "pre-init",
    _pre_init_log,
)


def _post_init_log(self) -> None:
    logger.info(f"Initialized {self.__class__.__name__}")


post_init_log_cb = (
    "post-init",
    _post_init_log,
)


def _log_loss_value(self, loss: Any) -> Any:
    if self.config.loss_log_interval != 0 and self.global_step % self.config.loss_log_interval == 0:
        logger.info(f"Global Step: {self.global_step}/{self.training_horizon}, Loss: {loss}")
    return loss


post_loss_log_cb = ("post-loss", _log_loss_value)


def _log_callback_ordering(self) -> None:
    log_str = f"Callback methods for {self.__class__.__name__}:"

    wrapped_methods = set(event.split("-")[1] for event in self._get_all_callback_event_methods())
    for event in wrapped_methods:
        log_str += f"\n  {event}:"

        log_str += "\n\tPre-callbacks:"
        pre_cbs = [cb for cb in self._initialized_callbacks if cb.event == f"pre-{event}"]
        log_str += str(pre_cbs)

        log_str += "\n\tPost-callbacks:"
        post_cbs = [cb for cb in self._initialized_callbacks if cb.event == f"post-{event}"]
        log_str += str(post_cbs)

    logger.info(log_str)


post_init_callback_ordering_cb = ("post-init", _log_callback_ordering)
