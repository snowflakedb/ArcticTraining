ps aux | grep vllm | awk '{print $2}' | xargs kill -9
