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

import pytest
from parameterized import parameterized

from arctic_training.testing_utils import CaptureStd
from arctic_training.testing_utils import TestCasePlus
from arctic_training.testing_utils import execute_subprocess_async
from arctic_training.testing_utils import get_unique_port_number
from arctic_training.testing_utils import require_torch_multi_gpu
from arctic_training.testing_utils import torch_assert_close
from arctic_training.testing_utils import torch_assert_safetensors_close
from arctic_training.testing_utils import write_file
from arctic_training.utils import read_json_file

# XXX: need to create a tiny dataset for the tests
train_dataset = "HuggingFaceH4/ultrachat_200k:train[:50]"

# qwen3 models: tiny random and smallish non-random
model_qwen = "snake7gun/tiny-random-qwen3moe"
# model_qwen = "DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder"

model_qwen_next = "yujiepan/qwen3-next-moe-tiny-random"

# gpt-oss models: tiny random and smallish non-random
model_gpt_oss = "tiny-random/gpt-oss-bf16"
# model_gpt_oss = "AmanPriyanshu/gpt-oss-6.0b-specialized-all-pruned-moe-only-7-experts"
# model_gpt_oss = "TroyDoesAI/gpt-oss-4B"

# XXX: todo next and gpt_oss
# models = [model_gpt_oss, model_qwen, model_qwen_next]

models = [model_qwen, model_qwen_next]


@pytest.mark.gpu
@require_torch_multi_gpu
class TestTrainerWithLauncher(TestCasePlus):
    # def setUp(self):
    #     super().setUp()

    @parameterized.expand(models)
    def test_moe(self, model_name_or_path):
        """compare ds+z2 vs ds+z2+amoe"""
        world_size = 2
        # later add support for pytest-xdist for unique ports
        master_port = get_unique_port_number()

        # XXX: fix this when done debugging
        shared_output_dir = self.get_auto_remove_tmp_dir("./xxx", after=False)
        shared_save_path = shared_output_dir / "saved"

        shared_config = f"""
type: sft
micro_batch_size: 1

model:
  name_or_path: {model_name_or_path}
  attn_implementation: flash_attention_2

deepspeed:
  zero_optimization:
    stage: 2

data:
  type: sft
  sources:
    - {train_dataset}
  cache_dir: {shared_save_path}/data-cache
  num_proc: 1
  dl_num_workers: 1

  max_length: 1024

logger:
  level: WARNING

epochs: 1

train_log_iter_interval: 1
seed: 42

"""

        # 1. e2e ds z3 baseline run
        baseline_output_dir = shared_output_dir / "baseline"
        baseline_output_dir.mkdir(exist_ok=True)
        baseline_log_train_file = baseline_output_dir / "train_logs.jsonl"
        baseline_log_train_file.unlink(missing_ok=True)
        baseline_checkpoint_dir = baseline_output_dir / "checkpoint"

        baseline_config = f"""
exit_iteration: 4

optimizer:
  type: fused_adam
  learning_rate: 1e-5

train_log_metrics_path: {baseline_log_train_file}

checkpoint:
  - type: huggingface
    save_end_of_training: true
    output_dir: {baseline_checkpoint_dir}

"""
        baseline_config_file = baseline_output_dir / "config.yaml"
        write_file(baseline_config_file, shared_config + baseline_config)
        cmd = f"""
            python -m arctic_training_cli {baseline_config_file} --num_gpus {world_size} --master_port {master_port}
            """.split()

        # keep for quick debug
        print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd))
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        self.assertIn("iter: 4/4", cs.combined)

        try:
            train_logs = read_json_file(baseline_log_train_file)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Error caught while reading {baseline_log_train_file}: {e}Relevant stderr output:\n{cs.err}"
            )
        self.assertEqual(train_logs[0]["iter"], 1)
        loss_a_1 = train_logs[0]["loss"]
        loss_a_2 = train_logs[1]["loss"]
        # loss_a_3 = train_logs[2]["loss"]

        # exit(0)

        # 2. e2e ds z2 arctic moe run
        amoe_output_dir = shared_output_dir / "amoe"
        amoe_output_dir.mkdir(exist_ok=True)
        amoe_log_train_file = amoe_output_dir / "train_logs.jsonl"
        amoe_log_train_file.unlink(missing_ok=True)
        amoe_checkpoint_dir = amoe_output_dir / "checkpoint"

        amoe_config = f"""
exit_iteration: 4

optimizer:
  type: fused_adam_moe
  learning_rate: 1e-5

arctic_moe: true
expert_parallel_size: 2

train_log_metrics_path: {amoe_log_train_file}

checkpoint:
  - type: huggingface
    save_end_of_training: true
    output_dir: {amoe_checkpoint_dir}
"""

        amoe_config_file = amoe_output_dir / "config.yaml"
        write_file(amoe_config_file, shared_config + amoe_config)
        cmd = f"""
            python -m arctic_training_cli {amoe_config_file} --num_gpus {world_size} --master_port {master_port}
            """.split()

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd))
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        self.assertIn("iter: 4/4", cs.combined)

        try:
            train_logs = read_json_file(amoe_log_train_file)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Error caught while reading {amoe_log_train_file}: {e}Relevant stderr output:\n{cs.err}"
            )
        self.assertEqual(train_logs[0]["iter"], 1)
        loss_b_1 = train_logs[0]["loss"]
        loss_b_2 = train_logs[1]["loss"]
        # loss_b_3 = train_logs[2]["loss"]

        # quality checks that loss is closely matching DS Z2 non-AMoE setup
        # dtype is lost when logs are saved, so matching againt bf16 standard torch.testing.assert_close tolerance which is rtol=1.6e-2, atol=1e-5
        torch_assert_close(loss_a_1, loss_b_1, rtol=1.6e-2, atol=1e-5)
        torch_assert_close(loss_a_2, loss_b_2, rtol=1.6e-2, atol=1e-5)
        # torch_assert_close(loss_a_3, loss_b_3, rtol=1.6e-2, atol=1e-5)

        # now check the conversion from AMoE to original model format worked correctly
        torch_assert_safetensors_close(
            baseline_checkpoint_dir / "global_step_4/model.safetensors",
            amoe_checkpoint_dir / "global_step_4/model.safetensors",
            rtol=1.6e-2,
            atol=1e-5,
        )
