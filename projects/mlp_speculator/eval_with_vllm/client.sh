#LLama 3 8B
#python3 benchmark_serving.py --backend vllm  --request-rate 2.0  --base-url http://127.0.0.1:8000 --dataset-path=/code/users/samyam/ShareGPT_Vicuna_unfiltered/ShareGPT_2023.05.04v0_Wasteland_Edition.json --dataset-name=sharegpt     --model NousResearch/Meta-Llama-3-8B-Instruct   --seed 12345


#Neural Magic Llama 3 70B
#python3 benchmark_serving.py --backend vllm  --request-rate 2.0  --base-url http://127.0.0.1:8000 --dataset-path=/code/users/samyam/snowflakedb-vllm/vllm/benchmarks/ShareGPT.json --dataset-name=sharegpt --model NousResearch/Meta-Llama-3.1-8B-Instruct --seed 12345

#Neural Magic Llama 3 70B
python3 benchmark_serving.py --backend vllm  --request-rate 2.0  --base-url http://0.0.0.0:8000 --dataset-path=/home/yak/jaeseong/ShareGPT.json --dataset-name=sharegpt --model /home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct --seed 12345

export request_rate=0.25
python /code/users/yewang/vllm_mlp_opt/benchmarks/benchmark_serving.py --backend vllm  --request-rate ${request_rate} --base-url http://127.0.0.1:8000 --dataset-path=/code/users/yewang/ShareGPT_2023.05.04v0_Wasteland_Edition.json --dataset-name=sharegpt --model /checkpoint/tmp/huggingface/llama33_swiftkv  --seed 12345 --num-prompts 300
python /code/users/yewang/vllm_mlp_opt/benchmarks/benchmark_serving.py --backend vllm  --request-rate ${request_rate} --base-url http://127.0.0.1:8000 --dataset-path=/code/users/yewang/ShareGPT_2023.05.04v0_Wasteland_Edition.json --dataset-name=sharegpt --model /checkpoint/tmp/huggingface/llama33_swiftkv  --seed 12345 --num-prompts 300
