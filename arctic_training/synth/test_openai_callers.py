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
