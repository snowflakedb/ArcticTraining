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

from tests.utils import run_dummy_training

dataset = "manu/project_gutenberg:en[:3]"
model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

model_config = dict(
    type="random-weight-hf",
    name_or_path=model_name,
)

data_config_training = dict(
    max_length=2048,
    sources=[
        dict(
            type="CausalDataset",
            name_or_path=dataset,
        )
    ],
)
data_config_evaluation = dict(
    max_length=2048,
    sources=[
        dict(
            type="CausalDataset",
            name_or_path=dataset,
        )
    ],
    train_eval_split=[0.8, 0.2],
)


@pytest.mark.parametrize(
    "run_on_cpu",
    [
        True,
        pytest.param(False, marks=pytest.mark.gpu),
    ],
)
def test_causal_trainer_training(model_name, run_on_cpu):
    run_dummy_training(
        {
            "type": "causal",
            "model": model_config,
            "data": data_config_training,
        },
        run_on_cpu=run_on_cpu,
    )


@pytest.mark.parametrize(
    "run_on_cpu",
    [
        True,
        pytest.param(False, marks=pytest.mark.gpu),
    ],
)
def test_causal_trainer_evaluation(model_name, run_on_cpu):
    trainer = run_dummy_training(
        {
            "type": "causal",
            "eval_interval": 1,
            "model": model_config,
            "data": data_config_evaluation,
        },
        run_on_cpu=run_on_cpu,
    )

    assert "loss/eval" in trainer.metrics.summary_dict, "loss/eval should be recorded in summary_dict"
    assert trainer.metrics.summary_dict["loss/eval"] > 0, "Evaluation should be greater than 0"
    assert "loss/eval" not in trainer.metrics.values, "loss/eval should not be recorded in values as it was logged"


@pytest.mark.parametrize(
    "run_on_cpu",
    [
        True,
        pytest.param(False, marks=pytest.mark.gpu),
    ],
)
def test_causal_trainer_evaluation_log_intervals(model_name, run_on_cpu):
    trainer = run_dummy_training(
        {
            "type": "causal",
            "exit_iteration": 3,
            "eval_interval": 1,
            "eval_log_iter_interval": 2,
            "model": model_config,
            "data": data_config_evaluation,
        },
        run_on_cpu=run_on_cpu,
    )

    assert "loss/eval" in trainer.metrics.values, "loss/eval should be recorded in values as `eval_interval` > 0"
    assert (
        "loss/eval" not in trainer.metrics.summary_dict
    ), "loss/eval should not be recorded in summary_dict as it should not be logged"
