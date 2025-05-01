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

import copy
import os
import types
from typing import Optional
from typing import Union

import torch
import torch.distributed as dist
from deepspeed.runtime.zero import GatheredParameters
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from speculator.configs import ArcticLSTMSpeculatorConfig
from speculator.speculator import ArcticLSTMSpeculator
from torch.nn import CrossEntropyLoss

from arctic_training import CheckpointEngine
from arctic_training import HFModelFactory
from arctic_training import ModelConfig
from arctic_training import SFTTrainer
from arctic_training import logger
from arctic_training.checkpoint.ds_engine import DSCheckpointEngine
from arctic_training.data.sft_factory import SFTDataFactory
from arctic_training.trainer.sft_trainer import to_device


class ArcticLSTMSpeculatorModelConfig(ModelConfig):
    n_speculator_heads: int = 3
    speculator_width: Union[int, str] = "4096"
    proj_dim: Union[int, str] = "4096"
    emb_dim: Union[int, str] = "4096"
    speculator_tie_weights: bool = False
    speculator_scale_input: bool = False
    method: str = "sum_rnn"
    tie_lstm_embs: bool = False


class ArcticLSTMSpeculatorModelFactory(HFModelFactory):
    name = "arctic-lstm-speculator"
    config: ArcticLSTMSpeculatorModelConfig

    def post_create_model_callback(self, model):
        hidden_size = model.lm_head.in_features
        vocab_size = model.lm_head.out_features

        speculator_config = ArcticLSTMSpeculatorConfig(
            self.config.name_or_path,
            hidden_size,
            self.config.speculator_width,
            self.config.proj_dim,
            self.config.emb_dim,
            vocab_size,
            self.config.n_speculator_heads,
            tie_weights=self.config.speculator_tie_weights,
            scale_input=self.config.speculator_scale_input,
            method=self.config.method,
            tie_lstm_embs=self.config.tie_lstm_embs,
        )

        model.speculator = ArcticLSTMSpeculator(speculator_config)

        model.speculator.to(model.dtype).to(model.device)

        model.speculator.reset_parameters()

        model.old_forward = model.forward

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            speculator_return: bool = False,
        ):
            """Forward pass of the SpeculatorModel.
            Returns:
                torch.Tensor: A tensor containing predictions from all Medusa heads.
                (Optional) Original predictions from the base model's LM head.
            """

            if not speculator_return:
                return self.old_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

            # Pass input through the base model
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            return outputs

        model.forward = types.MethodType(forward, model)

        if self.config.n_speculator_heads > 0:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.speculator.parameters():
                param.requires_grad = True

        if not self.config.disable_activation_checkpoint:
            model.gradient_checkpointing_enable()
            model = self.make_model_gradient_checkpointing_compatible(model)

        return model


class ArcticLSTMSpeculatorCheckpointEngine(CheckpointEngine):
    name = "arctic-lstm-speculator"

    def load(self, model) -> None:
        if dist.get_rank() == 0:
            load_path = os.path.join(self.config.checkpoint_dir, "pytorch_model.bin")
            state_dict = torch.load(load_path)

        is_z3 = self.trainer.model.zero_optimization_stage() == 3
        dist.barrier()

        assert len(state_dict.keys()) == len(
            model.speculator.named_paramters().keys()
        ), "Checkpoint has different parameters than the module"

        for name, param in model.speculator.named_parameters():
            if is_z3 and hasattr(param, "ds_id"):
                with GatheredParameters([param], modifier_rank=0):
                    if dist.get_rank() == 0:
                        param.copy_(state_dict[name])

    def save(self, model) -> None:
        if dist.get_rank() == 0:
            model_config = copy.deepcopy(model.speculator.config)
            model_to_save = ArcticLSTMSpeculator(model_config)
            parameters_to_save = model_to_save.parameters()
        else:
            parameters_to_save = [None for param in model.speculator.parameters()]

        is_z3 = self.trainer.model.zero_optimization_stage() == 3

        torch.cuda.empty_cache()
        dist.barrier()

        # Gather final model.
        for parameter_to_save, (name, ds_param) in zip(parameters_to_save, model.speculator.named_parameters()):
            # Using gathered parameter does not work.
            # Parameters tracking is messed up at this point
            # So we need to be selective when partitioning
            # This should oes not affect correctness.
            if is_z3 and hasattr(ds_param, "ds_id"):
                ds_param.all_gather(param_list=[ds_param])
                if dist.get_rank() == 0:
                    parameter_to_save.data.copy_(ds_param.data)
                if not ds_param.ds_active_sub_modules and ds_param.ds_status is not ZeroParamStatus.INFLIGHT:
                    ds_param.partition(param_list=[ds_param])
            else:
                if dist.get_rank() == 0:
                    parameter_to_save.data.copy_(ds_param.data)

        logger.info(f"Model saving at {self.config.output_dir}")
        if dist.get_rank() == 0 and self.config.output_dir is not None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            save_path = os.path.join(self.config.output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), save_path)
            model_config.save(self.config.output_dir)

        dist.barrier()


class ArcticLSTMSpeculatorTrainer(SFTTrainer):
    name = "arctic-lstm-speculator"
    data_factory: SFTDataFactory
    model_factory: ArcticLSTMSpeculatorModelFactory
    checkpoint_engine: Union[DSCheckpointEngine, ArcticLSTMSpeculatorCheckpointEngine]

    def loss(self, batch) -> float:
        inputs = to_device(batch, self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, speculator_return=True)
            hidden_states = outputs[0]  # b n h

        preds = self.model.speculator(
            hidden_states.detach()[:, : -self.model.speculator.n_predict - 1, :],
            inputs["input_ids"][:, 1:],
        )
        losses = []
        loss_fn = CrossEntropyLoss()

        labels = inputs["labels"]
        for i in range(preds.size(0)):
            targ = labels[:, i + 2 : preds.size(2) + i + 2]  # b n
            loss = loss_fn(preds[i].reshape(-1, preds.size(3)), targ.long().reshape(-1))
            losses.append(loss)

        loss = sum(losses)
        return loss
