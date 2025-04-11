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

import argparse
import datetime
import os

import yaml
from tqdm import tqdm

from arctic_training.synth.openai_callers import AzureOpenAISynth
from arctic_training.synth.openai_callers import OpenAISynth


def main():
    parser = argparse.ArgumentParser(description="Arctic Synth (Azure) OpenAI batch API command line tool.")

    parser.add_argument(
        "-c",
        "--credential",
        help="Credential file path.",
        type=str,
        default="~/.arctic_synth/credentials/default.yaml",
    )
    parser.add_argument("-w", "--work_dir", help="Work directory.", type=str, default="./batch_work_dir")
    parser.add_argument("-t", "--task", help="Task name.", type=str)
    parser.add_argument("-u", "--upload", help="Upload task to (Azure) OpenAI.", action="store_true")
    parser.add_argument("-s", "--submit", help="Submit task to (Azure) OpenAI.", action="store_true")
    parser.add_argument(
        "-r",
        "--retrieve",
        help="Retrieve task status from (Azure) OpenAI.",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--download",
        help="Download task from (Azure) OpenAI.",
        action="store_true",
    )
    parser.add_argument(
        "--clean_files_older_than_n_days",
        help="Clean files older than n days from (Azure) OpenAI.",
        type=int,
    )

    args = parser.parse_args()

    args.credential = os.path.expanduser(args.credential)
    if not os.path.exists(args.credential):
        raise FileNotFoundError(
            f"Credential file not found: {args.credential}. Please initialize the"
            " credential file first in your code or create one manually."
        )
    credential = yaml.safe_load(open(args.credential))
    if "azure_endpoint" in credential:
        client_class = AzureOpenAISynth
    else:
        client_class = OpenAISynth

    client = client_class(work_dir=args.work_dir, credential_path=args.credential)

    if args.task is None and any([args.upload, args.submit, args.retrieve, args.download]):
        raise ValueError("Task name is required. Use -t or --task to specify the task name.")

    if args.upload:
        client.upload_batch_task(args.task)
    if args.submit:
        client.submit_batch_task(args.task)
    if args.retrieve:
        client.retrieve_batch_task(args.task)
    if args.download:
        client.download_batch_task(args.task)
    if args.clean_files_older_than_n_days:
        files_to_delete = []
        continued_from = None
        for purpose in ["batch", "batch_output"]:  # compatibility due to API change
            while True:  # This loop is to handle pagination
                outdated_files = []
                files = client.files.list(purpose=purpose, extra_query={"order": "asc", "after": continued_from})
                outdated_files = [
                    f
                    for f in files.data
                    if (datetime.datetime.now() - datetime.datetime.fromtimestamp(f.created_at)).days
                    > args.clean_files_older_than_n_days
                ]
                files_to_delete.extend(outdated_files)
                if len(outdated_files) < len(files.data):  # We have collected all outdated files
                    break
                else:
                    continued_from = files.data[-1].id  # Continue from the last file
        file_ids_to_delete = [f.id for f in files_to_delete]
        outdated_files_count = len(file_ids_to_delete)
        if outdated_files_count == 0:
            print("No files to delete.")
        else:
            print(file_ids_to_delete)
            choice = input(
                f"Do you want to delete {outdated_files_count} files that are older"
                f" than {args.clean_files_older_than_n_days} days? (y/N) "
            )
            if choice.strip().lower() in ["y", "yes"]:
                for file_id in tqdm(file_ids_to_delete):
                    client.files.delete(file_id)
                print(f"{outdated_files_count} files deleted successfully.")
            else:
                print("No files deleted.")
    if not any(
        [
            args.upload,
            args.submit,
            args.retrieve,
            args.download,
            args.clean_files_older_than_n_days,
        ]
    ):
        parser.print_help()


if __name__ == "__main__":
    main()
