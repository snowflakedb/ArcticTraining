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

from arctic_training.testing_utils import CaptureStd
from arctic_training.testing_utils import TestCasePlus
from arctic_training.testing_utils import execute_subprocess_async
from arctic_training.testing_utils import get_unique_port_number
from arctic_training.testing_utils import write_file
from arctic_training.utils import read_json_file

# XXX: need to create a tiny dataset for the tests
train_dataset = "HuggingFaceH4/ultrachat_200k:train[:50]"
model_name_or_path = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.mark.gpu
class TestTrainerWithLauncher(TestCasePlus):

    def test_autocast(self):
        world_size = 1
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

  bf16:
    enabled: true
    bf16_master_weights_and_grads: true
    bf16_optimizer_states: true

  torch_autocast:
    enabled: true
    dtype: torch.bfloat16

optimizer:
  learning_rate: 1e-5

model:
  name_or_path: {model_name_or_path}
  attn_implementation: sdpa

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
        self.assertLess(loss_a, 20)  # check the loss is there
