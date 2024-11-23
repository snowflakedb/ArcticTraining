import os
os.environ["HF_HOME"] = "/checkpoint/huggingface/hub/"

import tempfile
tempfile.tempdir = "/data-fast/temp"
print(tempfile.gettempdir())

from arctic_training.config import ModelConfig
from arctic_training.config import Config, DataConfig
from mlp_speculator.mlp_speculator_trainer import MLPSpeculatorTrainConfig
from mlp_speculator.mlp_speculator_trainer import MLPSpeculatorTrainer

if __name__ == "__main__":
    """
    --model_path ${BASE} \
    --tokenizer_path ${BASE} \
    --datasets HuggingFaceH4/ultrachat_200k teknium/OpenHermes-2.5 \
    --data_cache_dir /data-fast/sft-test \
    --output_dir ${FINAL}
    """

    model_path = "/checkpoint/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f"

    data_config = DataConfig(
        tokenizer=model_path,
        datasets=["HuggingFaceH4/ultrachat_200k"],
        use_data_cache=True,
        always_max_length=True,
        cache_processed_data=True,
        data_cache_dir="/data-fast/st-data-new",
        num_proc=16,
        max_length=4096,
    )

    model_config = ModelConfig(
        name_or_path=model_path,
        use_liger_kernel=False,
        disable_activation_checkpoint=True,
    )

    config = MLPSpeculatorTrainConfig(
        speculator_width=4096,
        n_speculator_heads=1,
        speculator_tie_weights=False,
        speculator_scale_input=False,
        gen_train=True,
        gen_micro_batch=384,
        gen_seq_length=64, #256,
        gen_prompt_length=64,
        gen_train_micro_batch=32,
        lr=1e-3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        deepspeed={"zero_optimization": {
        "stage": 2, 
        "stage3_param_persistence_threshold": 1.000000e+04, 
        "stage3_max_live_parameters": 3.000000e+07, 
        "stage3_prefetch_bucket_size": 3.000000e+07, 
        "memory_efficient_linear": False
    }}, 
        gradient_accumulation_steps=8,
        betas=(0.9, 0.999),
        seed=42,
        epochs=5,
        micro_batch_size=6,
        data=data_config,
        model=model_config,
        checkpoint={"type":"mlp_speculator", "output_dir":"/data-fast/debug", "save_every_n_steps":1, "save_every_n_epochs":1},
    )

    trainer = MLPSpeculatorTrainer(config)
    trainer.train()

