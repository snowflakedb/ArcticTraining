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

"""
This example shows how to use the Arctic Embed codebase to finetune
the venerable E5-base-v2 model (released in May 2023) on a version of MSMARCO
training data which has been hard-negative-mined using a more modern technique.

The code needed to recreate the training data can be found in the sibling directory
`data_prep` within the `hard_negative_mining` subdirectory.

Original model paper: https://arxiv.org/abs/2212.03533
Model page: https://huggingface.co/intfloat/e5-base-v2
Better negative mining paper: https://arxiv.org/abs/2407.15831
"""
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

from arctic_embed.biencoder_model_factory import BiencoderModelConfig
from arctic_embed.contrastive_dataloader import ContrastivePretokenizedDataConfig
from arctic_embed.core.cuda_allocator_config import CUDA_ALLOCATOR_CONFIG_FOR_DYNAMICALLY_SIZED_DATA
from arctic_embed.trainer import BiencoderTrainer
from arctic_embed.trainer import BiencoderTrainerConfig

from arctic_training.config.checkpoint import CheckpointConfig
from arctic_training.config.logger import LoggerConfig
from arctic_training.config.optimizer import OptimizerConfig
from arctic_training.config.wandb import WandBConfig
from arctic_training.scheduler.wsd_factory import WSDSchedulerConfig

LEARNING_RATE = 3e-5
GRADIENT_CLIPPING = 10.0
# DATA_PATH = str(Path(__file__).parent / "data" / "pretrain_amazonqa" / "batched_16384")
DATA_PATH = (
    "s3://ml-dev-sfc-or-dev-misc1-k8s/cortexsearch/biencoder/pretrain_data_arctic_training_format/Qwen_Qwen3_Embedding_0.6B/combined_all_16384"
)
# EVAL_DATA_PATHS = [str(path) for path in (Path(__file__).parent / "data" / "eval").iterdir() if path.is_dir()]  # fix this
datasets = [
    "amazon_qa",
    "ccnews_de_v1",
    "ccnews_en_v1",
    "ccnews_es_v1",
    "ccnews_fr_v1",
    "ccnews_it_v1",
    "ccnews_pl_v1",
    "ccnews_pt_v1",
    "faq",
    "mc4_de_v1",
    "mc4_en_v1",
    "mc4_es_v1",
    "mc4_fr_v1",
    "mc4_it_v1",
    "mc4_pl_v1",
    "mc4_pt_v1",
    "mwiki_de_v1",
    "mwiki_en_v1",
    "mwiki_es_v1",
    "mwiki_fr_v1",
    "mwiki_it_v1",
    "mwiki_pl_v1",
    "mwiki_pt_v1",
    "paq",
    "pes2o",
    "red_pajama",
    "red_pajamas_1t_stackexchange",
    "s2orc_title_abstracts",
    "snippets4",
    "techrepo",
    "top_stories",
    "trivia_qa",
    "wikipedia",
]
EVAL_DATA_PATHS = [
    f"s3://ml-dev-sfc-or-dev-misc1-k8s/cortexsearch/biencoder/pretrain_data_arctic_training_format/Qwen_Qwen3_Embedding_0.6B/combined_all_16384_eval/{dataset}"
    for dataset in datasets
]
# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained("BAAI/bge-m3-retromae")
# tok.pad_token_id  --> 1
PAD_VALUE = 151643
LEFT_PAD = True


def now_timestamp_str() -> str:
    """Get the current ISO 8601 UTC timestamp."""
    return datetime.now(timezone.utc).strftime(r"%Y%m%dT%H%M%SZ")


ts = now_timestamp_str()
checkpoint_dir = Path(__file__).parent / "checkpoints" / "pretrain_qwen3_0point6" / ts
mconf = BiencoderModelConfig(
    name_or_path="Qwen/Qwen3-0.6B",
    pooling="last_token",
    disable_activation_checkpoint=False,
)
dconf = ContrastivePretokenizedDataConfig(
    filesystem="s3",
    root_directory=DATA_PATH,
    # filesystem="local",
    # root_directory=DATA_PATH,
    # Depending on how much GPU memory you have, you may need to split each
    # batch into a number of smaller sub-batches by setting the split_factor.
    # If you do so, you will probably want to decrease the learning rate accordingly.
    # split_factor=4,
    max_seq_length_query=32,
    max_seq_length_doc=256,
    eval_root_directories=EVAL_DATA_PATHS,
    eval_max_seq_length_doc=32,
    eval_max_seq_length_query=256,
    pad_value=PAD_VALUE,
    left_pad=LEFT_PAD,
)
sconf = WSDSchedulerConfig(num_warmup_steps=2000, num_decay_steps=2000)
oconf = OptimizerConfig(weight_decay=0.01, learning_rate=LEARNING_RATE)
lconf = LoggerConfig(level="INFO")
wconf = WandBConfig(
    enable=True,
    project="arctic-training-arctic-embed-testbed",
    name=f"qwen3_0point6-pretrain-{ts}",
)
# Reference: https://www.deepspeed.ai/training/#gradient-clipping
dsconf = {
    "gradient_clipping": GRADIENT_CLIPPING,
    "zero_optimization": {"stage": 1},
    # NOTE: The underlying DeepSpeed engine scales gradients down by a factor of
    # `1/world_size`` in the backwards pass, so we pre-scale the loss up by a factor
    # of `world_size`. Given these scalings, there is a potential for increased
    # numerical imprecision when using low-precision floating point representation,
    # so we set communication to fp32 in the backwards all-reduce to somewhat mitigate
    # this risk.
    "communication_data_type": "fp32",
}
cconf = CheckpointConfig(
    output_dir=checkpoint_dir,
    type="biencoder",
    save_every_n_steps=300,
    save_end_of_training=True,
)


def configure_non_distributed_distributed_training_if_needed() -> None:
    """Detect if we need to manually initialize distributed training environment
    and do so if needed.

    NOTE: We have to do this step because Arctic Training doesn't have a default
    1-GPU launching mode and will instead fall back to trying to auto-discover
    distributed training configuration (e.g. via MPI).
    """
    num_cli_args = len(sys.argv) - 1
    if num_cli_args == 0:
        print("***No CLI args detected, configuring for single-GPU training.***")
        from os import environ

        from torch import distributed as dist

        environ["MASTER_ADDR"] = "localhost"
        environ["MASTER_PORT"] = "12335"
        environ["LOCAL_RANK"] = "0"
        dist.init_process_group(backend="nccl", world_size=1, rank=0)


if __name__ == "__main__":
    CUDA_ALLOCATOR_CONFIG_FOR_DYNAMICALLY_SIZED_DATA.set_env()
    configure_non_distributed_distributed_training_if_needed()
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
        loss_log_interval=0,
        eval_frequency=300,
        use_in_batch_negatives=True,
        loss_temperature=0.02,
        overfit_first_batch=False,
        mrl_dim=256,
    )
    trainer = BiencoderTrainer(config=tconf)
    trainer.train()
