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
This example shows basic usage of the Arctic Embed codebase.

To run: `deepspeed minimal_example.py`
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from arctic_embed.biencoder_model_factory import BiencoderModelConfig
from arctic_embed.contrastive_dataloader import ContrastivePretokenizedDataConfig
from arctic_embed.trainer import BiencoderTrainer
from arctic_embed.trainer import BiencoderTrainerConfig

from arctic_training.config.checkpoint import CheckpointConfig
from arctic_training.config.logger import LoggerConfig
from arctic_training.config.optimizer import OptimizerConfig
from arctic_training.scheduler.wsd_factory import WSDSchedulerConfig

# EXAMPLE DATA #
# fmt: off
Q1_TOKENS = [101, 7592, 2088, 102]  # "Hello world"
Q2_TOKENS = [101, 2038, 2696, 2474, 13005, 1010, 2088, 102]  # "Hasta la vista, world"
# "With the possible exception of the equator, everything begins somewhere." - C.S. Lewis  # noqa: E501
D1_TOKENS = [101, 2007, 1996, 2825, 6453, 1997, 1996, 26640, 1010, 2673, 4269, 4873, 1012, 102]  # noqa: E501
# "In the end, it's not the years in your life that count. It's the life in your years." - Abraham Lincoln  # noqa: E501
D2_TOKENS = [101, 1999, 1996, 2203, 1010, 2009, 1005, 1055, 2025, 1996, 2086, 1999, 2115, 2166, 2008, 4175, 1012, 2009, 1005, 1055, 1996, 2166, 1999, 2115, 2086, 1012, 102]  # noqa: E501
# fmt: on

# Write example data to disk as ten copies of the same batch.
table_query = pa.table(
    {"BATCH_QUERY_ID": [0, 1], "QUERY_TOKEN_ID_LIST": [Q1_TOKENS, Q2_TOKENS]},
)
table_document = pa.table(
    {"BATCH_DOCUMENT_ID": [10, 11], "DOCUMENT_TOKEN_ID_LIST": [D1_TOKENS, D2_TOKENS]},
)
table_relation = pa.table(
    {"BATCH_QUERY_ID": [0, 1], "BATCH_DOCUMENT_ID": [10, 11], "RELEVANCE": [1, 1]},
    # schema=pa.schema(
    #     {
    #         "BATCH_QUERY_ID": pa.uint64(),
    #         "BATCH_DOCUMENT_ID": pa.uint64(),
    #         "RELEVANCE": pa.int8(),
    #     }
    # ),
)
data_path = Path(__file__).parent / "example_data"
data_path.mkdir(exist_ok=True, parents=True)
for i in range(10):
    batch_dir = data_path / f"batch_{i}"
    batch_dir.mkdir(exist_ok=True)
    pq.write_table(table_query, batch_dir / "queries.parquet")
    pq.write_table(table_document, batch_dir / "documents.parquet")
    pq.write_table(table_relation, batch_dir / "relations.parquet")


# CONFIG #
checkpoint_dir = Path(__file__).parent / "checkpoints" / "basic_example"
mconf = BiencoderModelConfig(name_or_path="prajjwal1/bert-tiny", pooling="first_token")
dconf = ContrastivePretokenizedDataConfig(
    filesystem="local",
    root_directory=str(data_path),
)
sconf = WSDSchedulerConfig(num_warmup_steps=2, num_decay_steps=3)
oconf = OptimizerConfig(weight_decay=0.01, learning_rate=1e-5)
lconf = LoggerConfig(level="INFO")
dsconf = {"zero_optimization": {"stage": 1}}
cconf = CheckpointConfig(
    output_dir=checkpoint_dir,
    type="biencoder",
    save_every_n_steps=3,
    save_end_of_training=True,
)


if __name__ == "__main__":
    tconf = BiencoderTrainerConfig(
        type="biencoder",
        model=mconf,
        data=dconf,
        scheduler=sconf,
        optimizer=oconf,
        logger=lconf,
        checkpoint=cconf,
        deepspeed=dsconf,
        use_in_batch_negatives=True,
        loss_temperature=0.02,
    )
    trainer = BiencoderTrainer(config=tconf)
    trainer.train()
