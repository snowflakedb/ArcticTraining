from arctic_training.config.checkpoint import CheckpointConfig
from arctic_training.config.config import Config
from arctic_training.config.data import DataConfig
from arctic_training.config.model import ModelConfig
from arctic_training.trainer.sft_trainer import SFTTrainer

data_config = DataConfig(
    tokenizer="meta-llama/Meta-Llama-3.1-8B-Instruct",
    datasets=["HuggingFaceH4/ultrachat_200k"],
    use_data_cache=True,
    cache_processed_data=True,
    data_cache_dir="/data-fast/st-data-new",
    num_proc=16,
)
model_config = ModelConfig(
    name_or_path="meta-llama/Meta-Llama-3.1-8B",
    use_liger_kernel=True,
)
checkpoint_config = CheckpointConfig(
    type="deepspeed", save_every_n_steps=10, output_dir="/data-fast/checkpoint/"
)
config = Config(
    model=model_config,
    data=data_config,
    checkpoint=checkpoint_config,
    micro_batch_size=2,
    epochs=1,
    lr=1e-5,
    gradient_accumulation_steps=1,
)
print(config)
trainer = SFTTrainer(config)
trainer.train()
