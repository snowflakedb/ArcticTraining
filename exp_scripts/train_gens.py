# import os.path
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# for ultrachat_response_length, ratios in [
#     # (range(1, 8), [0]),
#     (range(1, 8), range(10)),
#     # (range(1, 8), range(4, 7)),
# ]:
#     DATASETS = []
#     for ratio in ratios:
#         DATASETS.append(f'"(JsonDataset,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_fixed/magicoder/01_{ratio}_messages.jsonl)"')
#         for url in ultrachat_response_length:
#             DATASETS.append(f'"(JsonDataset,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_fixed/ultrachat/{url:02d}_{ratio}_messages.jsonl)"')
#     DATASETS = ' '.join(DATASETS)
#     OUTPUT_PATH = f"url_{'.'.join([str(url) for url in ultrachat_response_length])}_ratio_{'.'.join([str(ratio) for ratio in ratios])}"
#
#     TRAIN_ARGS=f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 2 --global_batch_size 128 --n_speculator_heads 3"
#     scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
#     """
#     with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#         f.write(scripts)
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# for ultrachat_response_length, ratios in [
#     # (range(1, 8), [0]),
#     (range(1, 8), range(10)),
#     # (range(1, 8), range(4, 7)),
# ]:
#     DATASETS = []
#     for ratio in ratios:
#         DATASETS.append(f'"(JsonInputOutput,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_fixed/magicoder/01_{ratio}_input_output.jsonl)"')
#         for url in ultrachat_response_length:
#             DATASETS.append(f'"(JsonInputOutput,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_fixed/ultrachat/{url:02d}_{ratio}_input_output.jsonl)"')
#     DATASETS = ' '.join(DATASETS)
#     OUTPUT_PATH = f"url_{'.'.join([str(url) for url in ultrachat_response_length])}_ratio_{'.'.join([str(ratio) for ratio in ratios])}_input_output"
#
#     TRAIN_ARGS=f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 2 --global_batch_size 128 --n_speculator_heads 3"
#     scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
#     """
#     with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#         f.write(scripts)
#
#
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# for ultrachat_response_length, ratios in [
#     # (range(1, 8), [0]),
#     (range(1, 8), range(10)),
#     # (range(1, 8), range(4, 7)),
# ]:
#     DATASETS = []
#     for ratio in ratios:
#         DATASETS.append(f'"(JsonInputOutput,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_fixed/magicoder/01_{ratio}_input_output.jsonl)"')
#         for url in ultrachat_response_length:
#             DATASETS.append(f'"(JsonInputOutput,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_fixed/ultrachat/{url:02d}_{ratio}_input_output.jsonl)"')
#     DATASETS = ' '.join(DATASETS)
#     OUTPUT_PATH = f"mask_url_{'.'.join([str(url) for url in ultrachat_response_length])}_ratio_{'.'.join([str(ratio) for ratio in ratios])}_input_output"
#
#     TRAIN_ARGS=f"--mask_inputs --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 2 --global_batch_size 128 --n_speculator_heads 3"
#     scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
#     """
#     with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#         f.write(scripts)
#
#
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# for ultrachat_response_length, ratios in [
#     # (range(1, 8), [0]),
#     (range(1, 8), range(10)),
#     # (range(1, 8), range(4, 7)),
# ]:
#     DATASETS = []
#     DATASETS = ' '.join(DATASETS)
#     OUTPUT_PATH = f"ultrachat_magicoder"
#
#     TRAIN_ARGS=f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 2 --global_batch_size 128 --n_speculator_heads 3"
#     scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
#     """
#     with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#         f.write(scripts)
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# DATASETS = []
# total_splits = 16
# for split in range(total_splits):
#     DATASETS.append(f'"(JsonOutput,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nostop/ultrachat/{split}/{split}_{total_splits}.jsonl)"')
#     DATASETS.append(f'"(JsonOutput,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nostop/magicoder/{split}/{split}_{total_splits}.jsonl)"')
# DATASETS = ' '.join(DATASETS)
# OUTPUT_PATH = f"256_chunks_ultrachat_magicoder_nostop"
#
# TRAIN_ARGS=f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 2 --global_batch_size 128 --n_speculator_heads 3"
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#     f.write(scripts)
#
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# DATASETS = []
# total_splits = 8
# for split in range(total_splits):
#     DATASETS.append(f'"(JsonOutput,/home/yak/jaeseong/ArcticTraining/toks/input_output_{split}.json)"')
# DATASETS = ' '.join(DATASETS)
# OUTPUT_PATH = f"generated_output_only"
#
# TRAIN_ARGS=f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 2 --global_batch_size 128 --n_speculator_heads 3"
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#     f.write(scripts)
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# DATASETS = []
# total_splits = 8
# for split in range(total_splits):
#     DATASETS.append(f'"(JsonInputOutput,/home/yak/jaeseong/ArcticTraining/toks/input_output_{split}.json)"')
# DATASETS = ' '.join(DATASETS)
# OUTPUT_PATH = f"generated_input_output"
#
# TRAIN_ARGS=f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 2 --global_batch_size 128 --n_speculator_heads 3"
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#     f.write(scripts)
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# DATASETS = []
# total_splits = 8
# for split in range(total_splits):
#     DATASETS.append(f'"(JsonCleverInputOutput,/home/yak/jaeseong/ArcticTraining/toks/input_output_{split}.json)"')
# DATASETS = ' '.join(DATASETS)
# OUTPUT_PATH = f"clever_generated"
#
# TRAIN_ARGS=f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=320 --global_batch_size 2048 --n_speculator_heads 3 --sim_gen_loss --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#     f.write(scripts)
#
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# DATASETS = []
# DATASETS.append(f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/toks_rawoutput)"')
# DATASETS = ' '.join(DATASETS)
# OUTPUT_PATH = f"raw_output"
#
# TRAIN_ARGS=f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#     f.write(scripts)
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# DATASETS = []
# DATASETS.append(f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/toks_rawinputoutput)"')
# DATASETS = ' '.join(DATASETS)
# OUTPUT_PATH = f"raw_inputoutput"
#
# TRAIN_ARGS=f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=320 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#     f.write(scripts)
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# DATASETS = []
# DATASETS.append(f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"')
# DATASETS = ' '.join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk"
#
# TRAIN_ARGS=f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#     f.write(scripts)
#
# MODEL_PATH="/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS=3000
#
# DATASETS = []
# DATASETS.append(f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"')
# DATASETS = ' '.join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_aurickloss"
#
# TRAIN_ARGS=f"--aurick_loss --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#     f.write(scripts)
#
# DATASETS = []
# DATASETS.append(f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"')
# DATASETS = ' '.join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_ctcloss_1"
#
# TRAIN_ARGS=f"--ctc_loss_weight=1 --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# aws s3 sync s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas/llama3.1_gen_mlpspec_nodetok_disk /home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", 'w') as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_spec_tie_weight"
# TRAIN_ARGS = f"--speculator_tie_weights --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_spec_scale_input"
# TRAIN_ARGS = f"--speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_spec_scale_input_tie_weight"
# TRAIN_ARGS = f"--speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = 8192
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}"
# TRAIN_ARGS = f"--speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = 8192 * 2
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}"
# TRAIN_ARGS = f"--speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_spec_scale_input_tie_weight_5head"
# TRAIN_ARGS = f"--speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 5 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 1000
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_1head"
# TRAIN_ARGS = f"--model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 1 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 1000
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
#
# for num_heads in range(2, 6):
#     OUTPUT_PATH = f"vllm_nodetok_disk_{num_heads}head"
#     TRAIN_ARGS = f"--checkpoint_path vllm_nodetok_disk_{num_heads-1}head --freeze_layers='{'|'.join([str(s) for s in range(num_heads - 1)])}' --auto_resume --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads {num_heads} --not_packing_input "
#     scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
#     """
#     with open(f"{OUTPUT_PATH}.sh", "w") as f:
#         f.write(scripts)
#
#
# for num_heads in range(2, 6):
#     OUTPUT_PATH = f"vllm_nodetok_disk_{num_heads}head_nofreeze"
#     TRAIN_ARGS = f"--checkpoint_path vllm_nodetok_disk_{num_heads-1}head --auto_resume --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads {num_heads} --not_packing_input "
#     scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
#     """
#     with open(f"{OUTPUT_PATH}.sh", "w") as f:
#         f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_aurickloss_tied_scale"
#
# TRAIN_ARGS = f"--speculator_tie_weights --speculator_scale_input --loss_type=aurick_loss --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# loss_type = "detach_conditional"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_{loss_type}_tied_scale"
#
# TRAIN_ARGS = f"--speculator_tie_weights --speculator_scale_input --loss_type={loss_type} --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# loss_type = "detach_conditional"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_{loss_type}"
#
# TRAIN_ARGS = f"--loss_type={loss_type} --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# param_init_method = "from_model_else_ones"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tied_scale_{param_init_method}"
# TRAIN_ARGS = f"--param_init_method={param_init_method} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# param_init_method = "from_model_else_zeros"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tied_scale_{param_init_method}"
# TRAIN_ARGS = f"--param_init_method={param_init_method} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# param_init_method = "from_model_else_ones"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tied_scale_{param_init_method}_add_orig_state"
# TRAIN_ARGS = f"--freeze_layers lm_head --add_orig_state --param_init_method={param_init_method} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# # aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# param_init_method = "zeros"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tied_scale_{param_init_method}_add_orig_state"
# TRAIN_ARGS = f"--freeze_layers lm_head --add_orig_state --param_init_method={param_init_method} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# # aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# param_init_method = "from_model_else_ones"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tied_{param_init_method}"
# TRAIN_ARGS = f"--param_init_method={param_init_method} --speculator_tie_weights --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# param_init_method = "from_model_else_ones"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tied_{param_init_method}_relu"
# TRAIN_ARGS = f"--use_relu --param_init_method={param_init_method} --speculator_tie_weights --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width 4096 --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = 8192 * 4
# micro_batch_size = 32 // 2
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}"
# TRAIN_ARGS = f"--speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {micro_batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = 8192 * 8
# micro_batch_size = 32 // 4
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}"
# TRAIN_ARGS = f"--speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {micro_batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = "4096.4096"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}"
# TRAIN_ARGS = f"--speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = "4096.4096.4096"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}"
# TRAIN_ARGS = f"--speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = "4096.4096"
# emb_dim = "4096.4096"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}"
# TRAIN_ARGS = f"--emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = "4096.4096"
# proj_dim = "4096.4096"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_proj{proj_dim}"
# TRAIN_ARGS = f"--proj_dim={proj_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = "4096.4096"
# proj_dim = "4096.4096"
# emb_dim = "4096.4096"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}"
# )
# TRAIN_ARGS = f"--proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = "16384.16384"
# proj_dim = "16384"
# emb_dim = "16384"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}"
# )
# TRAIN_ARGS = f"--proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = "16384.8192"
# proj_dim = "16384"
# emb_dim = "16384"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}"
# )
# TRAIN_ARGS = f"--proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# mlp_hidden_dim = "8192.16384"
# proj_dim = "8192"
# emb_dim = "8192"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}"
# )
# TRAIN_ARGS = f"--proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# param_init_method = "from_model_else_ones"
# mlp_hidden_dim = "16384"
# proj_dim = "16384"
# emb_dim = "16384"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tied_scale_{param_init_method}_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}"
# TRAIN_ARGS = f"--param_init_method={param_init_method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# param_init_method = "from_model_else_ones"
# mlp_hidden_dim = "16384.16384"
# proj_dim = "16384"
# emb_dim = "16384"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tied_scale_{param_init_method}_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}"
# TRAIN_ARGS = f"--param_init_method={param_init_method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# loss_type = "detach_conditional"
# mlp_hidden_dim = "16384"
# proj_dim = "16384"
# emb_dim = "16384"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{loss_type}"
# TRAIN_ARGS = f"--loss_type={loss_type} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# method = "sum_lstm"
# mlp_hidden_dim = "4096"
# proj_dim = "4096"
# emb_dim = "4096"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# method = "sum_lstm"
# mlp_hidden_dim = "4096"
# proj_dim = "4096"
# emb_dim = "4096"
# n_heads = 5
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}"
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "4096"
# proj_dim = "4096"
# emb_dim = "4096"
# n_heads = 5
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}"
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# method = "sum_lstm"
# mlp_hidden_dim = "8192"
# proj_dim = "8192"
# emb_dim = "8192"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# method = "sum_lstm"
# mlp_hidden_dim = "16384"
# proj_dim = "16384"
# emb_dim = "16384"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# method = "sum_lstm"
# mlp_hidden_dim = "12288"
# proj_dim = "12288"
# emb_dim = "12288"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/38ff4e01a70559264c95945aa04b900a11e68422/"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "8192"
# proj_dim = "8192"
# emb_dim = "8192"
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"llama33_swiftkvgen_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/38ff4e01a70559264c95945aa04b900a11e68422/"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "16384"
# proj_dim = "16384"
# emb_dim = "16384"
# batch_size = 32
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"llama33_swiftkvgen_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/38ff4e01a70559264c95945aa04b900a11e68422/"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "16384.16384"
# proj_dim = "16384"
# emb_dim = "16384"
# batch_size = 32
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"llama33_swiftkvgen_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/checkpoint/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/38ff4e01a70559264c95945aa04b900a11e68422/"
# TRAIN_ITERS = 3000
# method = "sum_lstm"
# mlp_hidden_dim = "12288"
# proj_dim = "12288"
# emb_dim = "12288"
# batch_size = 32
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"llama33_swiftkvgen_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/38ff4e01a70559264c95945aa04b900a11e68422/"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "16384.16384"
# proj_dim = "16384"
# emb_dim = "16384"
# batch_size = 32
# param_init_method = "from_model_else_ones"
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"llama33_swiftkvgen_{param_init_method}_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}"
# TRAIN_ARGS = f"--method={method} --param_init_method={param_init_method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "8192"
# proj_dim = "8192"
# emb_dim = "8192"
# batch_size = 32
# n_heads = 3
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "8192"
# proj_dim = "8192"
# emb_dim = "8192"
# batch_size = 32
# n_heads = 5
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# method = "sum_lstm"
# mlp_hidden_dim = "12288"
# proj_dim = "12288"
# emb_dim = "12288"
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_tie_embs"
# TRAIN_ARGS = f"--tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size 32 --max_length=256 --global_batch_size 2048 --n_speculator_heads 3 --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "12288"
# proj_dim = "12288"
# emb_dim = "12288"
# batch_size = 32
# n_heads = 3
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# method = "sum_lstm"
# mlp_hidden_dim = "8192"
# proj_dim = "8192"
# emb_dim = "8192"
# batch_size = 32
# n_heads = 5
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs"
# TRAIN_ARGS = f"--tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "4096"
# proj_dim = "4096"
# emb_dim = "4096"
# batch_size = 64
# n_heads = 3
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "6144"
# proj_dim = "6144"
# emb_dim = "6144"
# batch_size = 64
# n_heads = 3
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = (
#     f"swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}"
# )
# TRAIN_ARGS = f"--method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "6144"
# proj_dim = "6144"
# emb_dim = "6144"
# batch_size = 32
# n_heads = 3
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "8192"
# proj_dim = "8192"
# emb_dim = "8192"
# batch_size = 32
# n_heads = 3
# param_init_method = "from_model_else_ones"
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs_{param_init_method}"
# TRAIN_ARGS = f"--param_init_method={param_init_method} --checkpoint_path {OUTPUT_PATH}/checkpoints --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# method = "sum_rnn"
# mlp_hidden_dim = "6144"
# proj_dim = "6144"
# emb_dim = "6144"
# batch_size = 32
# n_heads = 5
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "8192"
# proj_dim = "8192"
# emb_dim = "8192"
# batch_size = 32
# n_heads = 5
# param_init_method = "zeros"
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs_{param_init_method}"
# TRAIN_ARGS = f"--param_init_method={param_init_method} --checkpoint_path {OUTPUT_PATH}/checkpoints --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "6144"
# proj_dim = "6144"
# emb_dim = "6144"
# batch_size = 32
# n_heads = 5
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# rm -r {OUTPUT_PATH}/checkpoints
# aws s3 sync {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{os.path.basename(OUTPUT_PATH)}
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "4096"
# proj_dim = "4096"
# emb_dim = "4096"
# batch_size = 32
# n_heads = 3
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# rm -r {OUTPUT_PATH}/checkpoints
# aws s3 sync {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{os.path.basename(OUTPUT_PATH)}
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "2048"
# proj_dim = "2048"
# emb_dim = "2048"
# batch_size = 64
# n_heads = 3
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# rm -r {OUTPUT_PATH}/checkpoints
# aws s3 sync {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{os.path.basename(OUTPUT_PATH)}
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "1024"
# proj_dim = "1024"
# emb_dim = "1024"
# batch_size = 64
# n_heads = 3
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# rm -r {OUTPUT_PATH}/checkpoints
# aws s3 sync {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{os.path.basename(OUTPUT_PATH)}
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# method = "sum_lstm"
# mlp_hidden_dim = "8192"
# proj_dim = "8192"
# emb_dim = "8192"
# batch_size = 4
# n_heads = 3
# max_len = 4096
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_len{max_len}_tie_embs"
# TRAIN_ARGS = f"--tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval 3000 --micro_batch_size {batch_size} --max_length={max_len} --global_batch_size {2048 * 256 // max_len} --n_speculator_heads {n_heads} "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# aws s3 cp {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{OUTPUT_PATH} --recursive
# """
# with open(f"{OUTPUT_PATH}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "6144"
# proj_dim = "6144"
# emb_dim = "6144"
# batch_size = 32
# n_heads = 4
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# rm -r {OUTPUT_PATH}/checkpoints
# aws s3 sync {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{os.path.basename(OUTPUT_PATH)}
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "6144"
# proj_dim = "6144"
# emb_dim = "6144"
# batch_size = 32
# n_heads = 2
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# rm -r {OUTPUT_PATH}/checkpoints
# aws s3 sync {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{os.path.basename(OUTPUT_PATH)}
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "6144"
# proj_dim = "6144"
# emb_dim = "6144"
# batch_size = 32
# n_heads = 1
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# rm -r {OUTPUT_PATH}/checkpoints
# aws s3 sync {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{os.path.basename(OUTPUT_PATH)}
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
#
# MODEL_PATH = "/home/yak/jaeseong/Meta-Llama-3.1-8B-Instruct"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "8192"
# proj_dim = "8192"
# emb_dim = "8192"
# batch_size = 32
# n_heads = 3
# max_len = 256
#
# DATASETS = []
# DATASETS.append(
#     f'"(RawDisk,/home/yak/jaeseong/ArcticTraining/llama3.1_gen_mlpspec_nodetok_disk)"'
# )
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.1-8b/jaeseong/vllm_nodetok_disk_tie_scale_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_len{max_len}_tie_embs_pasthidden"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --use_past_hidden_states --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length={max_len} --global_batch_size {2048 * 256 // max_len} --n_speculator_heads {n_heads} {'--not_packing_input' if max_len == 256 else ''} "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# rm -r {OUTPUT_PATH}/checkpoints
# aws s3 sync {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{os.path.basename(OUTPUT_PATH)}
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
#
# MODEL_PATH = "/checkpoint/tmp/huggingface/llama33_swiftkv_16bits/step-9722"
# TRAIN_ITERS = 3000
# check_interval = 300
# method = "sum_lstm"
# mlp_hidden_dim = "6144"
# proj_dim = "6144"
# emb_dim = "6144"
# batch_size = 64
# n_heads = 3
#
# DATASETS = []
# DATASETS.append(f'"\(RawDisk,/data/jaelee/swiftkv_llama33_gen_mlpspec_nodetok_disk\)"')
# DATASETS = " ".join(DATASETS)
# OUTPUT_PATH = f"/checkpoint/speculator/llama-3.3-70b/jaeseong/swiftkv_llama33_{mlp_hidden_dim}_emb{emb_dim}_proj{proj_dim}_{method}_{n_heads}_tie_embs_pasthidden"
# TRAIN_ARGS = f"--checkpoint_path {OUTPUT_PATH}/checkpoints --use_past_hidden_states --tie_lstm_embs --method={method} --proj_dim={proj_dim} --emb_dim={emb_dim} --speculator_tie_weights --speculator_scale_input --model_path {MODEL_PATH} --output_path {OUTPUT_PATH} --train_iters {TRAIN_ITERS} --datasets {DATASETS} --zero_stage 3 --speculator_width {mlp_hidden_dim} --checkpoint_interval {check_interval} --micro_batch_size {batch_size} --max_length=256 --global_batch_size 2048 --n_speculator_heads {n_heads} --not_packing_input "
# scripts = f"""
# mkdir -p {OUTPUT_PATH}
# deepspeed -H /data-fast/hostfile projects/mlp_speculator/train_speculator.py {TRAIN_ARGS} &> {OUTPUT_PATH}/train.log
# rm -r {OUTPUT_PATH}/checkpoints
# aws s3 sync {OUTPUT_PATH} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/spec/{os.path.basename(OUTPUT_PATH)}
# """
# with open(f"{os.path.basename(OUTPUT_PATH)}.sh", "w") as f:
#     f.write(scripts)
