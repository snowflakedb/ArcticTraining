import json
import os
from collections import defaultdict
from typing import Any

import jsonlines
from tqdm.auto import tqdm

from .utils import import_error
from .utils import pass_function

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

from .base_caller import BatchProcessor


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
        outputs = self.llm.chat(
            messages=conversations, sampling_params=self.sampling_params, use_tqdm=True
        )
        responses = []
        for request, output in zip(requests, outputs):
            res = {"custom_id": request["custom_id"], "response": output}
            responses.append(res)

        if self.work_dir is not None:
            with jsonlines.open(
                os.path.join(self.work_dir, task_name, "results.jsonl"), "w"
            ) as writer:
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
                        {"content": x.text, "role": "assistant"}
                        for x in response["response"].outputs
                    ],
                }
            )
        return extracted


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

    def add_chat_to_batch_task(
        self, task_name, model, messages, options={"temperature": 1, "top_p": 1}
    ):
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
            with jsonlines.open(
                os.path.join(self.work_dir, task_name, "results.jsonl"), "w"
            ) as writer:
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
                        {"content": x["messages"], "role": "assistant"}
                        for x in response["response"]["choices"]
                    ],
                }
            )
        return extracted
