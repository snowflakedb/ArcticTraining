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

import json
import os
from collections import defaultdict
from typing import Any

import jsonlines
from tqdm.auto import tqdm

from arctic_training.synth.base_caller import BatchProcessor
from arctic_training.synth.utils import import_error
from arctic_training.synth.utils import pass_function

try:
    from vllm import LLM
    from vllm import SamplingParams
except ImportError:
    LLM = import_error
    SamplingParams = pass_function

try:
    from snowflake import connector
except ImportError:
    connector = import_error


class InMemoryBatchProcessor(BatchProcessor):
    """
    An in-memory batch processor for non-OpenAI processors.
    """

    def __init__(self, work_dir: str | None = None):
        self.work_dir = work_dir
        self.requests: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def save_batch_task(self, task_name):
        if self.work_dir is None:
            raise ValueError("work_dir is not defined.")

        os.makedirs(os.path.join(self.work_dir, task_name, "requests"), exist_ok=True)
        with jsonlines.open(
            os.path.join(self.work_dir, task_name, "requests", f"{task_name}.jsonl"),
            "w",
        ) as writer:
            writer.write_all(self.requests[task_name])


class CortexSynth(InMemoryBatchProcessor):
    """
    Cortex Synthesizer. This class calls Snowflake Cortex complete service.
    """

    def __init__(
        self,
        connection_params,
        work_dir=None,
    ):
        super().__init__(work_dir=work_dir)
        self.conn = connector.connect(**connection_params)

    def __del__(self):
        if hasattr(self, "conn"):
            self.conn.close()

    def add_chat_to_batch_task(self, task_name, model, messages, options={"temperature": 1, "top_p": 1}):
        request_id = f"{task_name}_{len(self.requests[task_name])}"
        self.requests[task_name].append(
            {
                "custom_id": request_id,
                "model": model,
                "messages": messages,
                "options": options,
            }
        )

    def execute_batch_task(self, task_name):
        requests = self.requests[task_name]
        if self.work_dir is not None:
            self.save_batch_task(task_name)

        responses = []
        for request in tqdm(requests):
            sql = """
                SELECT SNOWFLAKE.CORTEX.COMPLETE(
                    %s,
                    PARSE_JSON(%s),
                    PARSE_JSON(%s)
                )
            """

            model = request["model"]
            messages = json.dumps(request["messages"])
            options = json.dumps(request["options"])

            cursor = self.conn.cursor()
            cursor.execute(sql, (model, messages, options))

            output = json.loads(cursor.fetchone()[0])

            responses.append({"custom_id": request["custom_id"], "response": output})

        if self.work_dir is not None:
            with jsonlines.open(os.path.join(self.work_dir, task_name, "results.jsonl"), "w") as writer:
                writer.write_all(responses)
        return responses

    @staticmethod
    def extract_messages_from_responses(responses):
        extracted = []
        for response in responses:
            extracted.append(
                {
                    "custom_id": response["custom_id"],
                    "choices": [
                        {"content": x["messages"], "role": "assistant"} for x in response["response"]["choices"]
                    ],
                }
            )
        return extracted
