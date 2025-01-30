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
import time

from arctic_training.synth import AzureOpenAISynth

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAISynth(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-07-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

client.add_chat_to_batch_task(
    task_name="test_task",
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "hello_world"},
    ],
)

client.add_chat_to_batch_task(
    task_name="test_task",
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "hello_world_2"},
    ],
)

client.add_chat_to_batch_task(
    task_name="test_task1",
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "hello_world"},
    ],
)

client.add_chat_to_batch_task(
    task_name="test_task1",
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "hello_world_2"},
    ],
)

print(client.execute_batch_task("test_task1"))

client.save_batch_task("test_task")
client.upload_batch_task("test_task")
client.submit_batch_task("test_task")
client.retrieve_uploaded_files("test_task")

# sleep 24h
time.sleep(24 * 60 * 60)

client.retrieve_batch_task("test_task")
client.download_batch_task("test_task")
