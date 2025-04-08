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

import asyncio
import json
import os

import aiohttp
import jsonlines
import msgspec

from arctic_training import logger
from arctic_training.synth.callers import InMemoryBatchProcessor
from arctic_training.synth.utils import import_error
from arctic_training.synth.utils import pass_function
from arctic_training.synth.utils import recursive_to_dict
from arctic_training.synth.vllm_utils import kill_processes
from arctic_training.synth.vllm_utils import launch_vllm_servers

try:
    from vllm import LLM
    from vllm import SamplingParams
except ImportError:
    LLM = import_error
    SamplingParams = pass_function


class VllmSynth(InMemoryBatchProcessor):
    """
    vLLM Synthesizer. This class initializes a local vLLM instance for fast batch inference. Currently, multi-node inference is not supported.
    """

    def __init__(
        self,
        model_params,
        sampling_params=SamplingParams(temperature=1.0),
        work_dir=None,
    ):
        super().__init__(work_dir=work_dir)
        self.llm = LLM(**model_params)
        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams(**sampling_params)
        self.sampling_params = sampling_params

    def add_chat_to_batch_task(self, task_name, messages):
        request_id = f"{task_name}_{len(self.requests[task_name])}"
        self.requests[task_name].append(
            {
                "custom_id": request_id,
                "messages": messages,
            }
        )

    def execute_batch_task(self, task_name):
        requests = self.requests[task_name]
        if self.work_dir is not None:
            self.save_batch_task(task_name)

        conversations = [request["messages"] for request in requests]
        outputs = self.llm.chat(messages=conversations, sampling_params=self.sampling_params, use_tqdm=True)
        responses = []
        for request, output in zip(requests, outputs):
            res = {
                "custom_id": request["custom_id"],
                "response": recursive_to_dict(output),
            }
            responses.append(res)
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
                    "choices": [{"content": x["text"], "role": "assistant"} for x in response["response"]["outputs"]],
                }
            )
        return extracted


class MultiReplicaVllmSynth(InMemoryBatchProcessor):
    """
    vLLM Synthesizer. This class initializes a local vLLM instance for fast batch inference. Currently, multi-node inference is not supported.
    """

    def __init__(
        self,
        model_params,
        sampling_params=SamplingParams(temperature=1.0),
        work_dir=None,
        tensor_parallel=1,
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    ):
        super().__init__(work_dir=work_dir)

        assert "model" in model_params, "No model supplied."

        # launch vllm servers
        self.model = model_params["model"]
        self.process_ids, self.vllm_urls = launch_vllm_servers(self.model, tensor_parallel, gpu_ids=gpu_ids)

        if isinstance(sampling_params, dict):
            sampling_params = SamplingParams(**sampling_params)
        self.sampling_params = sampling_params

    # Send a prompt to the VLLM server and get the response asynchronously
    async def generate_response(self, session, conversation, vllm_url):
        payload = {"model": self.model, "messages": conversation}

        payload = payload | msgspec.to_builtins(self.sampling_params)

        async with session.post(vllm_url, json=payload) as response:
            if response.status == 200:
                generated_text = ""
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    def remove_prefix(text: str, prefix: str) -> str:
                        if text.startswith(prefix):
                            return text[len(prefix) :]
                        return text

                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                    if not chunk == "[DONE]":
                        data = json.loads(chunk)
                        if data["choices"][0]["message"]["content"]:
                            generated_text += data["choices"][0]["message"]["content"]

                result = generated_text
                return result
            else:
                logger.info(f"Error: {response.status} - {await response.text()}")
                return ""

    def process_conversations_across_replicas(self, conversations):
        num_urls = len(self.vllm_urls)

        async def process():
            timeout = aiohttp.ClientTimeout(total=1000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = [
                    self.generate_response(session, conversation, self.vllm_urls[i % num_urls])
                    for i, conversation in enumerate(conversations)
                ]
                return await asyncio.gather(*tasks)

        return asyncio.run(process())

    def add_chat_to_batch_task(self, task_name, messages):
        request_id = f"{task_name}_{len(self.requests[task_name])}"
        self.requests[task_name].append(
            {
                "custom_id": request_id,
                "messages": messages,
            }
        )

    def execute_batch_task(self, task_name):
        requests = self.requests[task_name]
        if self.work_dir is not None:
            self.save_batch_task(task_name)

        conversations = [request["messages"] for request in requests]
        outputs = self.process_conversations_across_replicas(conversations)

        responses = []
        for request, output in zip(requests, outputs):
            res = {"custom_id": request["custom_id"], "response": output}
            responses.append(res)
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
                    "choices": [{"content": response["response"], "role": "assistant"}],
                }
            )
        return extracted

    def teardown(self):
        kill_processes(self.process_ids)
