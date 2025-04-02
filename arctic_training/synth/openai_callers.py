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
import logging
import os
import time
from collections import defaultdict
from typing import Any

import jsonlines
import yaml
from openai import AzureOpenAI
from openai import OpenAI
from tabulate import tabulate

from arctic_training.synth.base_caller import BatchProcessor


def load_credentials(credential_path):
    """
    load credential from local file
    """
    if not os.path.exists(credential_path):
        raise FileNotFoundError(f"Credential file not found: {credential_path}")
    with open(credential_path, "r") as f:
        kwargs = yaml.safe_load(f)
    return kwargs


class OpenAIBatchProcessor(BatchProcessor):
    """
    OpenAI Batch Processor. This class is inherited by :class:`~.OpenAISynth` and :class:`~.AzureOpenAISynth`. This class defines methods used to manage batch tasks.
    """

    def __init__(
        self,
        work_dir: str = "./batch_work_dir",
        credential_path: str | None = None,
        save_to_credential_file: str = "~/.arctic_synth/credentials/default.yaml",
        batch_size: int = 50_000,  # default batch size for OpenAI is 50,000,
        polling_interval: int = 30,  # (in secs) default polling interval to check task status
        *args,
        **kwargs,
    ):
        # set work directory
        self.work_dir = work_dir
        logging.info(f"Using work directory: {work_dir}")

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        # initialize batch task
        self.requests: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.batch_size = batch_size
        self.polling_interval = polling_interval

        # save credential for command line tool
        if credential_path is None:
            credential_file = os.path.expanduser(save_to_credential_file)
            if not os.path.exists(os.path.dirname(credential_file)):
                os.makedirs(os.path.dirname(credential_file))
            with open(credential_file, "w") as f:
                yaml.dump(kwargs, f)
            logging.info(f"Credential saved to: {credential_file}")

    def add_chat_to_batch_task(self, task_name, **kwargs):
        """
        Add a chat completion request to the batch task.
        """
        request_id = f"{task_name}_{len(self.requests[task_name])}"
        self.requests[task_name].append(
            {
                "custom_id": request_id,
                "method": "POST",
                "url": self.chat_api_url,
                "body": kwargs,
            }
        )

    def save_batch_task(self, task_name):
        """
        Save batch task to the work directory.
        """
        if len(self.requests[task_name]) == 0:
            logging.info(f"No requests found for task: {task_name}")
            return
        logging.info(f"Creating batch task: {task_name}")
        # partition requests into batches
        batches = [
            self.requests[task_name][i : i + self.batch_size]
            for i in range(0, len(self.requests[task_name]), self.batch_size)
        ]
        # save batches to jsonlines
        for i, batch in enumerate(batches):
            batch_file = os.path.join(self.work_dir, task_name, "requests", f"{task_name}_{i}.jsonl")
            if not os.path.exists(os.path.dirname(batch_file)):
                os.makedirs(os.path.dirname(batch_file))
            if os.path.exists(batch_file):
                logging.warning(f"Overwriting existing batch file: {batch_file}")
            with jsonlines.open(batch_file, "w") as writer:
                for request in batch:
                    writer.write(request)
        logging.info(f"Batch task saved to: {self.work_dir}/{task_name}")

    def upload_batch_task(self, task_name):
        """
        Upload batch task to (Azure) OpenAI Files API.
        """
        # get all batch files
        batch_files = [
            os.path.join(self.work_dir, task_name, "requests", f)
            for f in os.listdir(os.path.join(self.work_dir, task_name, "requests"))
        ]

        # upload each batch file
        openai_file_ids = []
        for batch_file in batch_files:
            openai_file = self.files.create(file=open(batch_file, "rb"), purpose="batch")
            openai_file_ids.append(openai_file.id)

        # save file ids to task
        with open(os.path.join(self.work_dir, task_name, "file_ids.txt"), "w") as f:
            f.write("\n".join(openai_file_ids))

        logging.info(f"Batch task uploaded to Azure OpenAI: {task_name}")

        return openai_file_ids

    def retrieve_uploaded_files(self, task_name=None, file_ids=None, print_status=True):
        """
        Retrieve uploaded files from (Azure) OpenAI Files API.
        """
        if file_ids is None:
            if not os.path.exists(os.path.join(self.work_dir, task_name, "file_ids.txt")):
                raise FileNotFoundError(
                    f"No file ids are found on disk for task: {task_name}. Please upload the batch task first."
                )
            with open(os.path.join(self.work_dir, task_name, "file_ids.txt"), "r") as f:
                file_ids = f.read().splitlines()

        file_status = []
        file_responses = []
        for file_id in file_ids:
            file_response = self.files.retrieve(file_id)
            file_responses.append(file_response)
            file_status.append((file_id, file_response.status))

        if print_status:
            print(
                tabulate(
                    [["File ID", "Status"]] + file_status,
                    headers="firstrow",
                    tablefmt="fancy_grid",
                )
            )
        return file_responses

    def submit_batch_task(self, task_name, file_ids=None):
        """
        Submit batch task to (Azure) OpenAI Batch API.
        """
        if file_ids is None:
            if not os.path.exists(os.path.join(self.work_dir, task_name, "file_ids.txt")):
                raise FileNotFoundError(
                    f"No file ids are passed or found on disk for task: {task_name}."
                    " Please upload the batch task first."
                )
            with open(os.path.join(self.work_dir, task_name, "file_ids.txt"), "r") as f:
                file_ids = f.read().splitlines()

        batch_ids = []
        for file_id in file_ids:
            batch_response = self.batches.create(
                input_file_id=file_id,
                endpoint=self.chat_api_url,  # Currently only chat completions are supported
                completion_window="24h",  # Currently only 24h is supported
            )
            batch_ids.append(batch_response.id)

        with open(os.path.join(self.work_dir, task_name, "batch_ids.txt"), "w") as f:
            f.write("\n".join(batch_ids))

        logging.info(f"Batch task submitted to Azure OpenAI: {task_name}")

        return batch_ids

    def retrieve_batch_task(self, task_name=None, batch_ids=None, print_status=True):
        """
        Retrieve batch task from (Azure) OpenAI Batch API.
        """
        if batch_ids is None:
            if task_name is None:
                raise ValueError("Either task_name or batch_ids must be provided.")
            if not os.path.exists(os.path.join(self.work_dir, task_name, "batch_ids.txt")):
                raise FileNotFoundError(
                    f"No batch ids are passed or found on disk for task: {task_name}."
                    " Please submit the batch task first."
                )
            with open(os.path.join(self.work_dir, task_name, "batch_ids.txt"), "r") as f:
                batch_ids = f.read().splitlines()

        batch_status = []
        batch_responses = []
        for batch_id in batch_ids:
            batch_response = self.batches.retrieve(batch_id)
            batch_responses.append(batch_response)
            created_at = batch_response.created_at  # timestamp in int
            created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created_at))
            batch_status.append((batch_id, created_at, batch_response.status))

        if print_status:
            print(
                tabulate(
                    [["Batch ID", "Created at", "Status"]] + batch_status,
                    headers="firstrow",
                    tablefmt="fancy_grid",
                )
            )
        return batch_responses

    def cancel_batch_task(self, task_name):
        """
        Cancel batch task from (Azure) OpenAI Batch API.
        """
        if not os.path.exists(os.path.join(self.work_dir, task_name, "batch_ids.txt")):
            raise FileNotFoundError(
                f"No batch ids are found on disk for task: {task_name}. Please submit the batch task first."
            )
        with open(os.path.join(self.work_dir, task_name, "batch_ids.txt"), "r") as f:
            batch_ids = f.read().splitlines()

        for batch_id in batch_ids:
            self.batches.cancel(batch_id)
            logging.info(f"Batch task cancelled: {batch_id}")

    def download_batch_task(self, task_name):
        """
        Download batch task results from (Azure) OpenAI Files API.
        """
        batch_responses = self.retrieve_batch_task(task_name=task_name)

        all_responses = ""
        all_batch_ready = True
        for batch_response in batch_responses:
            output_file_id = batch_response.output_file_id

            if output_file_id:
                file_response = self.files.content(output_file_id)
                raw_responses = file_response.text.strip()
                if not os.path.exists(os.path.join(self.work_dir, task_name, "downloads")):
                    os.makedirs(os.path.join(self.work_dir, task_name, "downloads"))
                with open(
                    os.path.join(
                        self.work_dir,
                        task_name,
                        "downloads",
                        f"{batch_response.id}.jsonl",
                    ),
                    "w",
                ) as f:
                    f.write(raw_responses)
                all_responses += raw_responses + "\n"
            else:
                all_batch_ready = False

        all_responses = all_responses.strip()
        if all_batch_ready:
            logging.info(f"Batch task downloaded: {task_name}")
            # sort by custom_id
            all_responses = sorted(all_responses.split("\n"), key=lambda x: json.loads(x)["custom_id"])
            with open(os.path.join(self.work_dir, task_name, "results.jsonl"), "w") as f:
                f.write("\n".join(all_responses))
        else:
            logging.warning(f"Not all batches are ready for download: {task_name}")

        return all_responses

    def execute_batch_task(self, task_name, print_status=False):
        """
        A synchronous method to execute the batch task. This method will block until the task is completed. The batch task is executed by sequentially saving, uploading, submitting, polling, and downloading.
        """
        self.save_batch_task(task_name)
        file_ids = self.upload_batch_task(task_name)

        # Polling mechanism for completion
        while True:
            file_responses = self.retrieve_uploaded_files(file_ids=file_ids, print_status=print_status)
            statuses = [response.status for response in file_responses]
            if all(status == "processed" for status in statuses):
                break
            else:
                for response in file_responses:
                    if response.status == "error":
                        raise ValueError(f"File processing failed for {response.id}: {response.status_details}")
            logging.info("Waiting for all files to upload...")
            time.sleep(self.polling_interval)

        batch_ids = self.submit_batch_task(task_name, file_ids=file_ids)

        # Polling mechanism for completion
        while True:
            batch_responses = self.retrieve_batch_task(batch_ids=batch_ids, print_status=print_status)
            statuses = [response.status for response in batch_responses]
            if all(status == "completed" for status in statuses):
                break
            else:
                for response in batch_responses:
                    if response.status == "failed":
                        raise ValueError(f"Batch processing failed for {response.id}: {response.errors.data}")
            logging.info("Waiting for all batches to complete...")
            time.sleep(self.polling_interval)

        # Download completed batches
        return self.download_batch_task(task_name)

    @staticmethod
    def extract_messages_from_responses(responses):
        """
        Extract the response from the response object.
        """
        extracted = []
        for response_str in responses:
            response = json.loads(response_str)
            extracted.append(
                {
                    "custom_id": response["custom_id"],
                    "choices": [x["message"] for x in response["response"]["body"]["choices"]],
                }
            )
        return extracted


class OpenAISynth(OpenAI, OpenAIBatchProcessor):
    """
    OpenAI Synthesizer. This class is a wrapper around OpenAI Python SDK. It manages batch processing by maintaing a work directory to store batch tasks and results. This class works with OpenAI platform (https://platform.openai.com/). Please refer to :class:`~.OpenAIBatchProcessor` for methods.
    """

    def __init__(
        self,
        work_dir: str = "./batch_work_dir",
        credential_path: str | None = None,
        save_to_credential_file: str = "~/.arctic_synth/credentials/default.yaml",
        batch_size: int = 50_000,  # default batch size for OpenAI is 50,000
        polling_interval: int = 30,  # (in secs) default polling interval to check task status
        *args,
        **kwargs,
    ):
        if credential_path is not None:
            kwargs = load_credentials(credential_path)

        OpenAIBatchProcessor.__init__(  # type: ignore
            self,
            work_dir=work_dir,
            credential_path=credential_path,
            save_to_credential_file=save_to_credential_file,
            batch_size=batch_size,
            polling_interval=polling_interval,
            *args,
            **kwargs,
        )
        OpenAI.__init__(self, *args, **kwargs)
        self.chat_api_url = "/v1/chat/completions"


class AzureOpenAISynth(AzureOpenAI, OpenAIBatchProcessor):
    """
    Azure OpenAI Synthesizer. This class is a wrapper around OpenAI Python SDK. It manages batch processing by maintaing a work directory to store batch tasks and results. This class works with Azure OpenAI (https://oai.azure.com/). Please refer to :class:`~.OpenAIBatchProcessor` for methods.
    """

    def __init__(
        self,
        work_dir: str = "./batch_work_dir",
        credential_path: str | None = None,
        save_to_credential_file: str = "~/.arctic_synth/credentials/default.yaml",
        batch_size: int = 100_000,  # default batch size for Azure OpenAI is 100,000
        polling_interval: int = 30,  # (in secs) default polling interval to check task status
        *args,
        **kwargs,
    ):
        if credential_path is not None:
            kwargs = load_credentials(credential_path)

        # config sanity check
        if kwargs.get("azure_deployment") is not None:
            raise ValueError(
                "Do not set client-wise azure_deployment. There is a known bug"
                " (https://github.com/openai/openai-python/issues/1397) that will cause"
                " URL errors for many APIs."
            )

        OpenAIBatchProcessor.__init__(  # type: ignore
            self,
            work_dir=work_dir,
            credential_path=credential_path,
            save_to_credential_file=save_to_credential_file,
            batch_size=batch_size,
            polling_interval=polling_interval,
            *args,
            **kwargs,
        )
        AzureOpenAI.__init__(self, *args, **kwargs)
        self.chat_api_url = "/chat/completions"
