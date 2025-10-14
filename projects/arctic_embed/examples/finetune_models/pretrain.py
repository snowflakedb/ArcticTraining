# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from datetime import datetime
from datetime import timezone
from pathlib import Path

from arctic_embed.biencoder_model_factory import BiencoderModelConfig
from arctic_embed.biencoder_s3_checkpoint import BiencoderS3CheckpointConfig
from arctic_embed.contrastive_dataloader import ContrastivePretokenizedDataConfig
from arctic_embed.core.cuda_allocator_config import CUDA_ALLOCATOR_CONFIG_FOR_DYNAMICALLY_SIZED_DATA
from arctic_embed.trainer import BiencoderTrainer
from arctic_embed.trainer import BiencoderTrainerConfig

from arctic_training.config.logger import LoggerConfig
from arctic_training.config.optimizer import OptimizerConfig
from arctic_training.config.wandb import WandBConfig
from arctic_training.scheduler.wsd_factory import WSDSchedulerConfig


def now_timestamp_str() -> str:
    return datetime.now(timezone.utc).strftime(r"%Y%m%dT%H%M%SZ")


def build_trainer_config_from_json(cfg: dict) -> BiencoderTrainerConfig:
    # Model
    mconf = BiencoderModelConfig(
        name_or_path=cfg["BASE_MODEL"],
        pooling=cfg.get("POOLING_METHOD", "last_token"),
        disable_activation_checkpoint=not cfg.get("ACTIVATION_CHECKPOINTING", False),
    )

    # Data
    dconf = ContrastivePretokenizedDataConfig(
        filesystem=cfg.get("FILE_SYSTEM", "s3"),
        root_directory=cfg["TRAINING_DATA_PATH"],
        eval_root_directories=cfg.get("EVALUATION_DATA_PATHS", []),
        max_seq_length_query=cfg.get("MAX_SEQ_LENGTH_QUERY", 32),
        max_seq_length_doc=cfg.get("MAX_SEQ_LENGTH_DOC", 256),
        eval_max_seq_length_query=cfg.get("MAX_SEQ_LENGTH_QUERY", 32),
        eval_max_seq_length_doc=cfg.get("MAX_SEQ_LENGTH_DOC", 256),
        pad_value=cfg["PAD_VALUE"],
        left_pad=cfg.get("LEFT_PADDING", False),
    )

    # Sched/optim/logging
    sconf = WSDSchedulerConfig(
        num_warmup_steps=cfg.get("WARMUP_STEPS", 2000),
        num_decay_steps=cfg.get("DECAY_STEPS", 2000),
    )
    oconf = OptimizerConfig(
        weight_decay=cfg.get("WEIGHT_DECAY", 0.01),
        learning_rate=cfg["LEARNING_RATE"],
    )
    lconf = LoggerConfig(level=cfg.get("LOG_LEVEL", "INFO"))

    # W&B
    wconf = WandBConfig(
        enable=cfg.get("ENABLE_WANDB", True),
        project=cfg.get("WANDB_PROJECT", "arctic-training-arctic-embed-testbed"),
        name=cfg.get("WANDB_RUN_NAME", f"arctic-embed-{now_timestamp_str()}"),
    )

    # DeepSpeed
    dsconf = {
        "gradient_clipping": cfg.get("GRADIENT_CLIPPING", 10.0),
        "zero_optimization": {"stage": int(cfg.get("ZERO_STAGE", 1))},
        "communication_data_type": cfg.get("COMMUNICATION_DATA_TYPE", "fp32"),
    }

    # S3 Checkpoint configuration (always required)
    # Local cache directory
    local_cache_dir = Path("/tmp") / "arctic_embed_checkpoints_cache"
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    cconf = BiencoderS3CheckpointConfig(
        output_dir=local_cache_dir,  # Used as staging directory
        s3_path=cfg.get("S3_CHECKPOINT_PATH"),
        local_cache_dir=str(local_cache_dir),
        max_local_checkpoints=cfg.get("MAX_LOCAL_CHECKPOINTS", 3),
        save_every_n_steps=cfg.get("SAVE_STEPS", 300),
        save_end_of_training=cfg.get("SAVE_END_OF_TRAINING", True),
        auto_resume=cfg.get("AUTO_RESUME", True),
    )

    # Trainer config
    tconf = BiencoderTrainerConfig(
        type="biencoder",
        model=mconf,
        data=dconf,
        scheduler=sconf,
        optimizer=oconf,
        logger=lconf,
        checkpoint=cconf,
        wandb=wconf,
        deepspeed=dsconf,
        loss_log_interval=cfg.get("LOSS_LOG_INTERVAL", 0),
        eval_frequency=cfg.get("EVAL_STEPS", 300),
        use_in_batch_negatives=cfg.get("IN_BATCH_NEGATIVES", True),
        loss_temperature=cfg.get("LOSS_TEMPERATURE", 0.02),
        overfit_first_batch=cfg.get("OVERFIT_FIRST_BATCH", False),
        mrl_dim=cfg.get("MRL_DIM", 256),
    )
    return tconf


def main() -> None:
    CUDA_ALLOCATOR_CONFIG_FOR_DYNAMICALLY_SIZED_DATA.set_env()

    parser = argparse.ArgumentParser(description="Arctic-Embed biencoder pretraining via JSON config")
    parser.add_argument("config_json", type=str, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config_json, "r") as f:
        cfg = json.load(f)

    tconf = build_trainer_config_from_json(cfg)
    trainer = BiencoderTrainer(config=tconf)
    trainer.train()


if __name__ == "__main__":
    main()
