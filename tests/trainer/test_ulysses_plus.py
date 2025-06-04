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

from arctic_training.testing_utils import CaptureStd
from arctic_training.testing_utils import TestCasePlus
from arctic_training.testing_utils import execute_subprocess_async
from arctic_training.testing_utils import get_unique_port_number
from arctic_training.testing_utils import require_torch_multi_gpu
from arctic_training.testing_utils import torch_assert_equal
from arctic_training.testing_utils import write_file
from arctic_training.utils import read_json_file

# XXX: need to create a tiny dataset for the tests
train_dataset = "HuggingFaceH4/ultrachat_200k:train[:100]"
model_name_or_path = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@require_torch_multi_gpu
class TestTrainerWithLauncher(TestCasePlus):
    # def setUp(self):
    #     super().setUp()

    def test_ulysses_plus_e2e(self):
        """
        This is an end-to-end test:
        1. runs 2 iterations for a baseline on 2 gpus sp=1, dp=2, gas=1
        2. runs 2 iterations for a baseline+ulysses plus features enabled on 2 gpus sp=2, dp=2, gas=2 (4 sub-iterations in total)
        3. compares that the loss is the same as both trainings have seen the exact same data once. The grads match is checked via loss, because the 2nd iteration will already have grads modified.
        """
        world_size = 2
        # later add support for pytest-xdist for unique ports
        master_port = get_unique_port_number()

        output_dir = self.get_auto_remove_tmp_dir()  # "./xxx", after=False)
        save_path = output_dir / "saved"

        baseline_config = f"""
type: sft
micro_batch_size: 1
exit_iteration: 2

deepspeed:
  zero_optimization:
    stage: 3
optimizer:
  learning_rate: 1e-5

model:
  #type: "liger"
  name_or_path: {model_name_or_path}
  attn_implementation: flash_attention_2
  #attn_implementation: sdpa

data:
  sources:
    - {train_dataset}
  cache_dir: {save_path}/data-cache
  num_proc: 1
  dl_num_workers: 1

  max_length: 512

logger:
  level: WARNING

train_log_iter_interval: 1
"""

        ulysses_plus_extra_config = f"""
activation_checkpoint_cpu_offload: true
tiled_mlp_compute: true
sequence_parallel_size: {world_size}
gradient_accumulation_steps: {world_size}
"""

        config_file = output_dir / "config.yaml"
        launcher = f"""
            python -m arctic_training_cli {config_file} --num_gpus {world_size} --master_port {master_port}
            """.split()
        cmd = launcher

        # 1. e2e baseline run
        log_train_file = save_path / "logs" / "train_logs-baseline.jsonl"
        log_config = f"""
train_log_metrics_path: {log_train_file}
"""
        config = baseline_config + log_config
        write_file(config_file, config)
        log_train_file.unlink(missing_ok=True)

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd)); die
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        self.assertIn("iter: 1/2", cs.combined)
        self.assertIn("iter: 2/2", cs.combined)

        try:
            train_logs = read_json_file(log_train_file)
        except FileNotFoundError as e:
            raise RuntimeError(f"Error caught while reading {log_train_file}: {e}Relevant stderr output:\n{cs.err}")
        # test that we run max_num_opt_steps_this_run=3 steps and not more
        self.assertEqual(train_logs[0]["iter"], 1)
        loss_a = train_logs[0]["loss"]

        # 2. e2e with UlyssesPlus enabled (all features)
        log_train_file = save_path / "logs" / "train_logs-ulysses-plus.jsonl"
        log_config = f"""
train_log_metrics_path: {log_train_file}
"""
        config = baseline_config + log_config + ulysses_plus_extra_config
        write_file(config_file, config)
        log_train_file.unlink(missing_ok=True)

        # keep for quick debug
        # print(" ".join([f"\nPYTHONPATH={self.src_dir_str}"] + cmd)); die
        with CaptureStd() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        self.assertIn("iter: 0/2", cs.combined)
        self.assertIn("iter: 1/2", cs.combined)
        self.assertIn("iter: 1/2", cs.combined)
        self.assertIn("iter: 2/2", cs.combined)

        try:
            train_logs = read_json_file(log_train_file)
        except FileNotFoundError as e:
            raise RuntimeError(f"Error caught while reading {log_train_file}: {e}Relevant stderr output:\n{cs.err}")
        # test that we run max_num_opt_steps_this_run=3 steps and not more
        self.assertEqual(train_logs[0]["iter"], 1)
        loss_b = train_logs[0]["loss"]

        torch_assert_equal(loss_a, loss_b)
