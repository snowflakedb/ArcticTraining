# ArcticSynth

ArcticSynth is a Python client for data synthesis in batch. It provides support for different services and abstract the complexity away. It provides functionalities to manage batch tasks, including adding tasks, saving, uploading, submitting, retrieving, and downloading batch tasks. For (Azure) OpenAI, it provides both asynchronous and synchronous functions for data synthesis in batch. You can choose to submit and monitor the task execution, or run and forget, as if you're using an online inference service but 50% cheaper and way faster. We'll walk you through it.

## Installation

ArcticSynth is offered as part of ArcticTraining. Installing ArcticTraining will also install the ArcticSynth CLI. You will only need ArcticSynth CLI if you're using (Azure) OpenAI services.

```bash
cd ArcticTraining

# If using OpenAI/Azure OpenAI
pip install -e .

# If using Snowflake Cortex
pip install -e '.[cortex]'

# If using vLLM
pip install -e '.[vllm]'
```

## Usage

### Supported Services

ArcticSynth currently provides the following classes to support their corresponding services:
- `OpenAISynth`: OpenAI (batch API)
- `AzureOpenAISynth`: Azure OpenAI (batch API)
- `CortexSynth`: Snowflake Cortex
- `VllmSynth`: vLLM (local batch inference)

### Initialization

To initialize the ArcticSynth client, you need to provide your API key and other necessary configurations. For example, with Azure OpenAI:

```python
from arctic_training.arctic_synth import AzureOpenAISynth
import os

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAISynth(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-07-01-preview",
    azure_endpoint="https://<your-endpoint-url>",
)
```
Optionally, you can set `batch_size` to reduce or increase how many requests per batch has. The default size is 100,000, which is maximum allowed by Azure OpenAI. However, if your request is large, for example, if it contains encoded images, you can set it to a smaller number to avoid each file to be too large. The partition of batch files and merging of downloaded result files will be taken care of automatically by ArcticSynth.

> [!NOTE]
> Configurations vary for different services and API providers. You can check [Snowflake Cortex example](./test_cortex_caller.py) and [vLLM example](./test_vllm_caller.py) and the corresponding classes.

### Adding Tasks

You can add chat tasks to a batch using the `add_chat_to_batch_task` method. The parameters are consistent with the original OpenAI API.

```python
client.add_chat_to_batch_task(
    task_name="test_task",
    model="<model-name>",
    messages=[
        {"role": "user", "content": "Hello world!"},
    ],
)
```
You can also set `work_dir` to a different path. Default is `./batch_work_dir`.

### Synchronous: Executing Tasks

For all services, we provide a synchronous method to execute the task. For (Azure) OpenAI, this method will submit the task to the service and wait for the result. For other services, it will run the task locally and return the result.

```python
execution_results = client.execute_batch_task("test_task")
```

You can parse the results with the `extract_messages_from_responses` function:
```python
parsed_results = client.extract_messages_from_responses(execution_results)
```

### Asynchronous: Saving, Uploading, and Submitting Tasks

> [!NOTE]
> This section is only applicable to (Azure) OpenAI.

As an alternative to synchronous method `execute_batch_task`, you can choose to submit the task to (Azure) OpenAI and download the results later. To do so, after adding tasks, you can save, upload, and submit them to (Azure) OpenAI.

```python
client.save_batch_task("test_task")
client.upload_batch_task("test_task")
client.submit_batch_task("test_task")
```

### Asynchronous: Retrieving and Downloading Tasks

> [!NOTE]
> This section is only applicable to (Azure) OpenAI.

You can retrieve the status of your batch tasks and download the results when they are ready.

```python
client.retrieve_batch_task("test_task")
client.download_batch_task("test_task")
```


## Command Line Interface (CLI)

> [!NOTE]
> This section is only applicable to (Azure) OpenAI.

ArcticSynth also provides a command line interface for managing batch tasks.

### Usage

```bash
arctic_synth -t <task_name> [options]
```

### Options

- `-c`, `--credential`: Credential file path (default: `~/.arctic_synth/credentials/default.yaml`). This will be auto-saved if you previously used ArcticSynth to add requests to a task. Normally you shouldn't worry about it.
- `-w`, `--work_dir`: Work directory (default: `./batch_work_dir`)
- `-t`, `--task`: Task name (required for most operations)
- `-u`, `--upload`: Upload task to Azure OpenAI
- `-s`, `--submit`: Submit task to Azure OpenAI
- `-r`, `--retrieve`: Retrieve task status from Azure OpenAI.
- `-d`, `--download`: Download task from Azure OpenAI. You'll find the downloaded files in `downloads` and a merged jsonl file `results.jsonl` in your task dir.
- `--clean_files_older_than_n_days`: Clean files older than n days in the (Azure) OpenAI file storage.

> [!WARNING]
> `--clean_files_older_than_n_days` will clean all files (not just yours) older than n days in the (Azure) OpenAI file storage. Use with extra caution.
