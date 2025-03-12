#Run from VLLM Benchmark Folder

#LLama 3 8B
#python3 benchmark_serving.py --backend vllm  --request-rate 2.0  --base-url http://127.0.0.1:8000 --dataset-path=/code/users/samyam/ShareGPT_Vicuna_unfiltered/ShareGPT_2023.05.04v0_Wasteland_Edition.json --dataset-name=sharegpt     --model NousResearch/Meta-Llama-3-8B-Instruct   --seed 12345


#Neural Magic Llama 3 70B
#python3 benchmark_serving.py --backend vllm  --request-rate 2.0  --base-url http://127.0.0.1:8000 --dataset-path=/code/users/samyam/snowflakedb-vllm/vllm/benchmarks/ShareGPT.json --dataset-name=sharegpt --model NousResearch/Meta-Llama-3.1-8B-Instruct --seed 12345

#Neural Magic Llama 3 70B
python3 benchmark_serving.py --backend vllm  --request-rate 2.0  --base-url http://127.0.0.1:8000 --dataset-path=/code/users/samyam/snowflakedb-vllm/vllm/benchmarks/ShareGPT.json --dataset-name=sharegpt --model neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 --seed 12345
