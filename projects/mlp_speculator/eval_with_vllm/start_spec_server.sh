#Trained on Ultra Chat Distribution
#vllm serve NousResearch/Meta-Llama-3-8B-Instruct --swap_space 16 --disable-log-requests --enable-chunked-prefill --speculative_model /checkpoint/speculator/llama-3-2.5k-iters-lr-1e3-bsz-500K/meta-llama-3-8B-nous-research/global_step-2501/speculator_only/ --use-v2-block-manager --num-speculative-tokens 3

#Trained on Generated Data
#vllm serve NousResearch/Meta-Llama-3.1-8B-Instruct --swap_space 16 --disable-log-requests --enable-chunked-prefill --speculative_model /checkpoint/speculator/llama-3.1-8b/Nov-28-6-43/ --use-v2-block-manager --num-speculative-tokens 3

#vllm serve NousResearch/Meta-Llama-3-8B-Instruct --swap_space 16 --disable-log-requests --enable-chunked-prefill --speculative_model  /checkpoint/speculator/llama-3b-all-data-sets-3500/global_step-3501/speculator_only/ --use-v2-block-manager --num-speculative-tokens 3

#Jan 27
#vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 --swap_space 16 --tensor-parallel-size 2 --disable-log-requests --enable-chunked-prefill --speculative_model  /checkpoint/speculator/llama-3.1-70b/5-heads/synth-data/Jan-27/ --use-v2-block-manager --num-speculative-tokens 5

#Feb 3
#vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 --swap_space 16 --tensor-parallel-size 2 --disable-log-requests --enable-chunked-prefill --speculative_model  /checkpoint/speculator/llama-3.1-70b/5-heads/synth-data/Feb-3/ --use-v2-block-manager --num-speculative-tokens 5

#Feb 5
if [ ! -d $1 ]
then
  aws s3 sync s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/$1 $1
fi
vllm serve /home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct --swap_space 16 --disable-log-requests --enable-chunked-prefill --speculative_model $1 --use-v2-block-manager --num-speculative-tokens $2 --gpu-memory-utilization 0.7

#vllm serve /home/yak/jaeseong/Llama-3.3-70B-Instruct --swap_space 16 --disable-log-requests --enable-chunked-prefill --speculative_model /checkpoint/speculator/llama-3.3-70b/5-heads/Feb-11 --use-v2-block-manager --num-speculative-tokens 5 --tensor-parallel-size 4

#vllm serve neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 --swap_space 16 --tensor-parallel-size 2 --disable-log-requests --enable-chunked-prefill --speculative_model  meta-llama/Llama-3.2-1B-Instruct --use-v2-block-manager --num-speculative-tokens 5


#vllm serve NousResearch/Meta-Llama-3-70B-Instruct --swap_space 16 --tensor-parallel-size 4 --enable-chunked-prefill --speculative_model /checkpoint/speculator-70b/llama-3-2.5k-iters-lr-1e3-bsz-500K/meta-llama-3-8B-nous-research/global_step-2400/speculator-only/ --use-v2-block-manager --num-speculative-tokens 2

#vllm serve neuralmagic/Meta-Llama-3-70B-Instruct-FP8 --disable-log-requests --swap_space 16 --tensor-parallel-size 2 --enable-chunked-prefill --speculative_model /checkpoint/speculator-70b/llama-3-2.5k-iters-lr-1e3-bsz-500K/meta-llama-3-8B-nous-research/global_step-2400/speculator-only/ --use-v2-block-manager --num-speculative-tokens 2
#vllm serve neuralmagic/Meta-Llama-3-70B-Instruct-FP8 --disable-log-requests --swap_space 16 --tensor-parallel-size 2 --speculative_model /checkpoint/speculator-70b/llama-3-2.5k-iters-lr-1e3-bsz-500K/meta-llama-3-8B-nous-research/global_step-2400/speculator-only/ --use-v2-block-manager --num-speculative-tokens 2

#vllm serve neuralmagic/Meta-Llama-3-70B-Instruct-FP8 --disable-log-requests --swap_space 16 --tensor-parallel-size 2 --enable-chunked-prefill --use-v2-block-manager



#python -m vllm.entrypoints.api_server --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8 --disable-log-requests --swap_space 16 --tensor-parallel-size 2 --enable-chunked-prefill --speculative_model /checkpoint/speculator-70b/llama-3-2.5k-iters-lr-1e3-bsz-500K/meta-llama-3-8B-nous-research/global_step-2400/speculator-only/ --use-v2-block-manager --num-speculative-tokens 2
#python -m vllm.entrypoints.api_server --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8 --disable-log-requests --swap_space 16 --tensor-parallel-size 2 --speculative_model /checkpoint/speculator-70b/llama-3-2.5k-iters-lr-1e3-bsz-500K/meta-llama-3-8B-nous-research/global_step-2400/speculator-only/ --use-v2-block-manager --num-speculative-tokens 2
#python -m vllm.entrypoints.api_server --model neuralmagic/Meta-Llama-3-70B-Instruct-FP8 --disable-log-requests --swap_space 16 --tensor-parallel-size 2 --use-v2-block-manager --enable-chunked-prefill

#vllm serve NousResearch/Meta-Llama-3-70B-Instruct --swap_space 16 --tensor-parallel-size 4 --speculative_model ibm-fms/llama3-70b-accelerator --use-v2-block-manager --num-speculative-tokens 2 --enable-chunked-prefill


#     cat <<EOF > ${benchmark_dir}/start_server.sh
# python -m vllm.entrypoints.api_server \
#     ${vllm_options} \
#     --disable-log-requests
# EOF
#     cat <<EOF > ${benchmark_dir}/start_openai_server.sh
# python -m vllm.entrypoints.openai.api_server \
#     ${vllm_options} \
#     --disable-log-requests
# EOF
