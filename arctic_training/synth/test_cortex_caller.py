from os import getenv

from arctic_training.synth import CortexSynth

connection_params = {
    "account": getenv("SNOWFLAKE_ACCOUNT"),
    "user": getenv("SNOWFLAKE_USER"),
    "role": getenv("SNOWFLAKE_ROLE"),
    "warehouse": getenv("SNOWFLAKE_WAREHOUSE"),
    "authenticator": "externalbrowser",
}

client = CortexSynth(connection_params=connection_params)

for i in range(3):
    client.add_chat_to_batch_task(
        task_name="test_task_cortex",
        model="llama3.2-1b",
        options={"temperature": 1, "top_p": 0.95},
        messages=[
            {"role": "user", "content": f"hello_world_{i}"},
        ],
    )

extracted_messages = client.extract_messages_from_responses(
    client.execute_batch_task("test_task_cortex")
)
print(extracted_messages)
