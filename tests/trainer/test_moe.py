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

from parameterized import parameterized

from arctic_training.testing_utils import CaptureStd
from arctic_training.testing_utils import TestCasePlus
from arctic_training.testing_utils import execute_subprocess_async
from arctic_training.testing_utils import get_unique_port_number
from arctic_training.testing_utils import require_torch_multi_gpu
from arctic_training.testing_utils import torch_assert_close
from arctic_training.testing_utils import write_file
from arctic_training.utils import read_json_file

# XXX: need to create a tiny dataset for the tests
train_dataset = "HuggingFaceH4/ultrachat_200k:train[:50]"

# qwen3 models: tiny random and smallish non-random
model_qwen = "snake7gun/tiny-random-qwen3moe"
# model_qwen = "DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder"

# gpt-oss models: tiny random and smallish non-random
model_gpt_oss = "tiny-random/gpt-oss-bf16"
# model_gpt_oss = "AmanPriyanshu/gpt-oss-6.0b-specialized-all-pruned-moe-only-7-experts"
# model_gpt_oss = "TroyDoesAI/gpt-oss-4B"

models = [model_gpt_oss, model_qwen]


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

        output_dir = self.get_auto_remove_tmp_dir("./xxx", after=False)
        save_path = output_dir / "saved"

        baseline_config = f"""
type: sft
micro_batch_size: 1
exit_iteration: 4


model:
  #type: "liger"
  name_or_path: {model_name_or_path}
  attn_implementation: flash_attention_2

data:
  type: sft
  sources:
    - {train_dataset}
  cache_dir: {save_path}/data-cache
  num_proc: 1
  dl_num_workers: 1

  max_length: 1024

logger:
  level: WARNING

epochs: 1

train_log_iter_interval: 1
seed: 42
"""

        config_file = output_dir / "config.yaml"
        launcher = f"""
            python -m arctic_training_cli {config_file} --num_gpus {world_size} --master_port {master_port}
            """.split()
        cmd = launcher

        # 1. e2e ds z3 baseline run
        deepspeed_zero_config_extra = """
deepspeed:
  zero_optimization:
    stage: 2
    offload_optimizer:
      device: cpu

optimizer:
  type: cpu_adam
  learning_rate: 1e-5

#arctic_moe: false
#expert_parallel_size: 1
"""
        log_train_file = save_path / "logs" / "train_logs-ds-baseline.jsonl"
        log_config = f"""
train_log_metrics_path: {log_train_file}
"""
        config = baseline_config + deepspeed_zero_config_extra + log_config
        write_file(config_file, config)
        log_train_file.unlink(missing_ok=True)

        # keep for quick debug
        print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd))
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        self.assertIn("iter: 1/4", cs.combined)
        self.assertIn("iter: 2/4", cs.combined)

        try:
            train_logs = read_json_file(log_train_file)
        except FileNotFoundError as e:
            raise RuntimeError(f"Error caught while reading {log_train_file}: {e}Relevant stderr output:\n{cs.err}")
        # test that we run max_num_opt_steps_this_run=3 steps and not more
        self.assertEqual(train_logs[0]["iter"], 1)
        loss_a = train_logs[0]["loss"]

        # exit(0)

        # 2. e2e ds z2 arctic moe run
        arctic_moe_config_extra = """
deepspeed:
  zero_optimization:
    stage: 2
    offload_optimizer:
      device: cpu

optimizer:
  type: cpu_adam_moe
  learning_rate: 1e-5

arctic_moe: true
expert_parallel_size: 2
"""

        log_train_file = save_path / "logs" / "train_logs-arctic-moe.jsonl"
        log_config = f"""
train_log_metrics_path: {log_train_file}
"""
        config = baseline_config + arctic_moe_config_extra + log_config
        write_file(config_file, config)
        log_train_file.unlink(missing_ok=True)

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd)); die
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        self.assertIn("iter: 1/4", cs.combined)
        self.assertIn("iter: 2/4", cs.combined)

        try:
            train_logs = read_json_file(log_train_file)
        except FileNotFoundError as e:
            raise RuntimeError(f"Error caught while reading {log_train_file}: {e}Relevant stderr output:\n{cs.err}")
        # test that we run max_num_opt_steps_this_run=3 steps and not more
        self.assertEqual(train_logs[0]["iter"], 1)
        loss_b = train_logs[0]["loss"]

        # comparisons
        torch_assert_close(loss_a, loss_b)  # , atol=0, rtol=0)
