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

print(
    client.extract_messages_from_responses(client.execute_batch_task("test_task_qwen"))
)
