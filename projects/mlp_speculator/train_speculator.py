import os
import tempfile
import pprint
import argparse

from mlp_speculator.configs import MLPSpeculatorTrainConfig
from mlp_speculator.mlp_speculator_trainer import MLPSpeculatorTrainer

from arctic_training.config import DataConfig
from arctic_training.config import ModelConfig
from arctic_training.logging import logger

from projects.swiftkv.llama_swiftkv import (
    register_auto as swiftkv_model_register,
)
swiftkv_model_register()

import torch
import torch.distributed as dist

os.environ["HF_HOME"] = "/checkpoint/huggingface/hub/"
tempfile.tempdir = "/data-fast/temp"
os.makedirs(tempfile.tempdir, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
def MLPSpeculatorParser():
    def parse_tuple_or_string(arg):
        try:
            # Convert a string like "(1,2)" to a tuple
            arg = arg.strip("()")
            return tuple(map(str.strip, arg.split(",")))
        except Exception as e:
            try:
                return str(arg)
            except Exception as ee:
                raise argparse.ArgumentTypeError(f"Invalid format: {arg}")    
            
    parser = argparse.ArgumentParser(description="MLP Speculator Configurations")
    group = parser.add_argument_group(description="MLP Speculator Configs")
    
    group.add_argument('--local_rank', type=int, help='gpu rank')
    
    group.add_argument("--model_path", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct")
    group.add_argument("--output_path", type=str, default="")
    group.add_argument("--checkpoint_path", type=str, default=None)
    
    group.add_argument("--datasets", 
                       type=parse_tuple_or_string, 
                       nargs="+", 
                       default=["HuggingFaceH4/ultrachat_200k","ise-uiuc/Magicoder-OSS-Instruct-75K"])
    
    # DataLoader micro batch size
    group.add_argument("--global_batch_size", type=int, default=48)
    group.add_argument("--micro_batch_size", type=int, default=6)
    
    #Total training iterations
    group.add_argument("--train_iters", type=int, default=1000)
    group.add_argument("--checkpoint_interval", type=int, default=300)
    
    group.add_argument("--zero_stage", type=int, default=2)
    
    group.add_argument("--speculator_width", type=int, default=4096)
    group.add_argument("--n_speculator_heads", type=int, default=3)

    group.add_argument("--speculator_tie_weights", action="store_true", default=False)
    group.add_argument("--speculator_scale_input", action="store_true", default=False)
    group.add_argument("--speculator_path", type=str, default=None)
    group.add_argument("--weighted_sum", action="store_true", default=False)

    ################### Arguments for generative training ###################

    # if true, the training will generate data for training the speculator
    group.add_argument("--gen_train", action="store_true", default=False)
    group.add_argument("--mask_inputs", action="store_true", default=False)
    group.add_argument("--gen_train_global_batch", type=int, default=2048)
    group.add_argument("--gen_train_micro_batch", type=int, default=32)
    group.add_argument("--gen_micro_batch", type=int, default=384)
    
    return parser.parse_args()

if __name__ == "__main__":
    args=MLPSpeculatorParser()
    
    #need this to get gpu count to calculate gas
    dist.init_process_group(backend="nccl")
    
    model_path = args.model_path
    
    data_config = DataConfig(
        tokenizer=model_path,
        datasets=args.datasets,
        use_data_cache=True,
        always_max_length=True,
        cache_processed_data=True,
        data_cache_dir="/data-fast/st-data-new",
        num_proc=16,
        max_length=4096,
        mask_inputs=args.mask_inputs
    )

    model_config = ModelConfig(
        name_or_path=model_path,
        use_liger_kernel=False,
        disable_activation_checkpoint=True,
    )
    
    if args.gen_train:
        gradient_accumulation_steps = args.gen_train_global_batch // \
                                    args.gen_train_micro_batch // \
                                    torch.distributed.get_world_size()
    else:
        gradient_accumulation_steps = args.global_batch_size // \
                                    args.micro_batch_size // \
                                    torch.distributed.get_world_size()
                                                                      
    config = MLPSpeculatorTrainConfig(
        speculator_width=args.speculator_width,
        n_speculator_heads=args.n_speculator_heads,
        speculator_tie_weights=args.speculator_tie_weights,
        speculator_scale_input=args.speculator_scale_input,
        weighted_sum=args.weighted_sum,
        gen_train=args.gen_train,
        gen_micro_batch=args.gen_micro_batch,
        gen_seq_length=256,
        gen_prompt_length=64,
        gen_train_micro_batch=args.gen_train_micro_batch,
        lr=1e-3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        deepspeed={
            "zero_optimization": {
                "stage": args.zero_stage,        
                "allgather_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1.000000e04,
                "stage3_max_live_parameters": 3.000000e07,
                "stage3_prefetch_bucket_size": 5e8,
                "reduce_bucket_size": 2.5e8, 
                "memory_efficient_linear": False,
            }
        },
        gradient_accumulation_steps=gradient_accumulation_steps,
        betas=(0.9, 0.999),
        seed=42,
        epochs=10,
        micro_batch_size=args.micro_batch_size,
        train_iters=args.train_iters,
        data=data_config,
        model=model_config,
        checkpoint=[{
            "type": "mlp_speculator",
            "output_dir": args.output_path,
            "save_every_n_steps": args.checkpoint_interval,
            "save_every_n_epochs": 1,
            "save_end_of_training": True,
            "auto_resume": False
        },{
            "type": "deepspeed",
            "output_dir": args.checkpoint_path,
            "save_every_n_steps": args.checkpoint_interval,
            "save_every_n_epochs": 1,
            "save_end_of_training": True,
            "auto_resume": True
        }]
    )
    
    logger.info(f"Config: {pprint.pformat(config,indent=1)}")
    trainer = MLPSpeculatorTrainer(config)
    #trainer.checkpoint_engines[0].save()
    trainer.train()
