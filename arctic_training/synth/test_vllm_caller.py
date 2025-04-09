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

from vllm import SamplingParams

from arctic_training.synth import VllmSynth

client = VllmSynth(
    model_params={"model": "Qwen/Qwen2.5-0.5B-Instruct"},
    sampling_params=SamplingParams(temperature=0),
)

for i in range(10):
    client.add_chat_to_batch_task(
        task_name="test_task_qwen",
        messages=[
            {"role": "user", "content": f"hello_world_{i}"},
        ],
    )

print(client.extract_messages_from_responses(client.execute_batch_task("test_task_qwen")))
