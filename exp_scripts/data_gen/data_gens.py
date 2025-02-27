# total_len = 28
# for i in range(total_len):
#     output_dir = f"llama3.1_gen_fixed/ultrachat/{i}"
#     script = f"""
# python projects/mlp_speculator/mlp_speculator/data_generation/data_generation_js.py --cur_split={i} --output_dataset_path={output_dir} --total_split={total_len}
# aws s3 cp {output_dir} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas/{output_dir} --recursive
#     """
#     with open(f"{i:02}.sh", 'w') as f:
#         f.write(script)
#
# total_len = 8
# for i in range(total_len):
#     output_dir = f"llama3.1_gen_fixed/magicoder/{i}"
#     script = f"""
# python projects/mlp_speculator/mlp_speculator/data_generation/data_generation_js.py --hf_dataset magicoder --cur_split={i} --output_dataset_path={output_dir} --total_split={total_len}
# aws s3 cp {output_dir} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas/{output_dir} --recursive
#     """
#     with open(f"{i:02}.sh", 'w') as f:
#         f.write(script)
#

# total_len = 16
# for i in range(total_len):
#     output_dir = f"llama3.1_gen_mlpspec_nostop/ultrachat/{i}"
#     script = f"""
# python projects/mlp_speculator/mlp_speculator/data_generation/data_generation_mlpspec.py --cur_split={i} --output_dataset_path={output_dir} --total_split={total_len}
# aws s3 syn {output_dir} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas/{output_dir}
#     """
#     with open(f"{i:02}.sh", 'w') as f:
#         f.write(script)
#
# total_len = 16
# for i in range(total_len):
#     output_dir = f"llama3.1_gen_mlpspec_nostop/magicoder/{i}"
#     script = f"""
# python projects/mlp_speculator/mlp_speculator/data_generation/data_generation_mlpspec.py --hf_dataset magicoder --cur_split={i} --output_dataset_path={output_dir} --total_split={total_len}
# aws s3 sync {output_dir} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas/{output_dir}
#     """
#     with open(f"magic_{i:02}.sh", 'w') as f:
#         f.write(script)

# total_len = 8
# for i in range(total_len):
#     output_dir = f"llama3.1_gen_mlpspec_nodetok/ultrachat/{i}"
#     script = f"""
# python projects/mlp_speculator/mlp_speculator/data_generation/data_generation_mlpspec_using_promptids.py --cur_split={i} --output_dataset_path={output_dir} --total_split={total_len}
# aws s3 syn {output_dir} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas/{output_dir}
#     """
#     with open(f"{i:02}.sh", 'w') as f:
#         f.write(script)
#
# total_len = 8
# for i in range(total_len):
#     output_dir = f"llama3.1_gen_mlpspec_nodetok/magicoder/{i}"
#     script = f"""
# python projects/mlp_speculator/mlp_speculator/data_generation/data_generation_mlpspec_using_promptids.py --hf_dataset magicoder --cur_split={i} --output_dataset_path={output_dir} --total_split={total_len}
# aws s3 sync {output_dir} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas/{output_dir}
#     """
#     with open(f"magic_{i:02}.sh", 'w') as f:
#         f.write(script)


# tokenizer_name = "../Llama-3.3-70B-Instruct"
# model_name = "../llama33_swiftkv"
# tensor_parallel = 1
# total_len = 16
# for i in range(total_len):
#     output_dir = f"swiftkv_llama33_gen_mlpspec_nodetok_hf/ultrachat/{i}"
#     script = f"""
# python projects/mlp_speculator/mlp_speculator/data_generation/data_generation_mlpspec_using_promptids.py --model={model_name} --tensor_parallel={tensor_parallel} --tokenizer={tokenizer_name} --cur_split={i} --output_dataset_path={output_dir} --total_split={total_len}
# aws s3 sync {output_dir} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas/{output_dir}
#     """
#     with open(f"{i:02}.sh", 'w') as f:
#         f.write(script)
#
# total_len = 16
# for i in range(total_len):
#     output_dir = f"swiftkv_llama33_gen_mlpspec_nodetok_hf/magicoder/{i}"
#     script = f"""
# python projects/mlp_speculator/mlp_speculator/data_generation/data_generation_mlpspec_using_promptids.py --hf_dataset magicoder --model={model_name} --tensor_parallel={tensor_parallel} --tokenizer={tokenizer_name} --cur_split={i} --output_dataset_path={output_dir} --total_split={total_len}
# aws s3 sync {output_dir} s3://ml-dev-sfc-or-dev-misc1-k8s/yak/users/jaelee/datas/{output_dir}
#     """
#     with open(f"magic_{i:02}.sh", 'w') as f:
#         f.write(script)
