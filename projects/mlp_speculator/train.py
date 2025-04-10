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
import sys
import types
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from mlp_speculator.configs import MLPSpeculatorConfig
from mlp_speculator.speculator import MLPSpeculator
from pydantic import model_validator
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import DynamicCache
from typing_extensions import Self

from arctic_training import CheckpointEngine
from arctic_training import HFModelFactory
from arctic_training import ModelConfig
from arctic_training import SFTTrainer
from arctic_training import TrainerConfig
from arctic_training import logger
from arctic_training.trainer.sft_trainer import to_device


class MLPSpeculatorTrainerConfig(TrainerConfig):
    speculator_path: Optional[str] = None
    gen_train: bool = False
    gen_train_simple: bool = False
    gen_micro_batch_size: int = 384
    gen_train_global_batch_size: int = 2048
    gen_train_micro_batch_size: int = 32
    gen_prompt_length: int = 64
    gen_seq_length: int = 256

    @model_validator(mode="after")
    def set_grad_accum_steps(self):
        if self.gen_train:
            self.gradient_accumulation_steps = (
                self.gen_train_global_batch_size // self.gen_train_micro_batch_size // self.world_size
            )
        else:
            self.gradient_accumulation_steps = self.global_batch_size // self.micro_batch_size // self.world_size
        self = self.build_deepspeed_config()
        return self

    @model_validator(mode="after")
    def check_gen_train(self) -> Self:
        if self.gen_train and not self.gen_train_simple:
            assert (
                self.gen_micro_batch_size % self.gen_train_micro_batch_size == 0
            ), "gen_micro_batch_size must be divisible by gen_train_micro_batch_size"
        elif self.gen_train:
            assert (
                self.gen_train_micro_batch_size == self.gen_micro_batch_size
            ), "gen_train_micro_batch_size must equal gen_micro_batch_size"
        return self


class MLPSpeculatorModelConfig(ModelConfig):
    n_speculator_heads: int = 3
    speculator_width: int = 4096
    speculator_tie_weights: bool = False
    speculator_scale_input: bool = False


class MLPSpeculatorModelFactory(HFModelFactory):
    name = "spec-decode"
    config: MLPSpeculatorModelConfig

    def post_create_model_callback(self, model):
        hidden_size = model.lm_head.in_features
        vocab_size = model.lm_head.out_features

        speculator_config = MLPSpeculatorConfig(
            self.config.name_or_path,
            hidden_size,
            self.config.speculator_width,
            vocab_size,
            self.config.n_speculator_heads,
            tie_weights=self.config.speculator_tie_weights,
            scale_input=self.config.speculator_scale_input,
        )

        model.speculator = MLPSpeculator(speculator_config)

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


class MLPSpeculatorCheckpointEngine(CheckpointEngine):
    name = "spec-decode"

    def load(self) -> None:
        raise NotImplementedError()

    def save(self) -> None:
        if dist.get_rank() == 0:
            model_config = copy.deepcopy(self.model.speculator.config)
            model_to_save = MLPSpeculator(model_config)
            parameters_to_save = model_to_save.parameters()
        else:
            parameters_to_save = [None for param in self.model.speculator.parameters()]

        is_z3 = self.trainer.model.zero_optimization_stage() == 3

        torch.cuda.empty_cache()
        dist.barrier()

        # Gather final model.
        for parameter_to_save, (name, ds_param) in zip(parameters_to_save, self.model.speculator.named_parameters()):
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


class MLPSpeculatorTrainer(SFTTrainer):
    name = "spec-decode"
    config: MLPSpeculatorTrainerConfig
    model_factory: MLPSpeculatorModelFactory
    checkpoint_engine: MLPSpeculatorCheckpointEngine

    def generate(
        self,
        inputs,
        max_seq_len: int = 2048,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 10,
        do_sample: bool = True,
        num_beams: int = 1,
        use_cache: bool = False,
        contiguous_cache: bool = False,
        include_embeds: bool = True,
    ):
        """
        A straightforward copy of the generate method in fms.utils.generation.
        The only change is the include_embeds flag, which when true also returns
        the embedding vectors corresponding to the tokens in the output sequence.

        Args:
            self : self is the trainer.
        """

        input_ids = inputs["input_ids"]
        assert type(input_ids) is torch.Tensor and input_ids.dim() == 2, "Invalid Input Shape. Must be b x n"

        embeds = None
        result = input_ids
        next_input = input_ids

        input_dict = dict()
        input_dict["past_key_values"] = DynamicCache()
        input_dict["use_cache"] = use_cache

        for _ in range(max_new_tokens):
            input_dict["input_ids"] = next_input[:, -max_seq_len:]
            output = self.model(**input_dict, speculator_return=True)
            hidden_states = output[0]
            if use_cache:
                past_key_values = output[1]
                input_dict["past_key_values"] = past_key_values
            logits = self.model.module.lm_head(hidden_states)
            logits = logits[:, -1, :]

            if do_sample:
                # get logits from last value in sequence nad scale
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")

                probs = F.softmax(logits, dim=-1)
                next_val = torch.multinomial(probs, num_samples=1)
            else:
                next_val = torch.argmax(logits, dim=-1).unsqueeze(0).t()

            result = torch.cat((result, next_val), dim=-1)

            if use_cache:
                next_input = next_val
            else:
                next_input = result

            if include_embeds:
                if embeds is None:
                    embeds = hidden_states
                else:
                    embeds = torch.cat((embeds, hidden_states), dim=-2)

        if include_embeds:
            return result, embeds

        return result

    def _compute_loss1(self, inputs):
        """
        Compute the training loss for the model.

        Args:
            inputs (dict): The input data, including input IDs, attention mask, and labels.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        inputs = to_device(inputs, self.device)

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

    def _compute_loss2(self, inputs):
        """
        Compute the training loss for the model.

        Args:
            inputs (dict): The input data, including input IDs, attention mask, and labels.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """

        config = self.config
        inputs = to_device(inputs, self.device)

        with torch.no_grad():
            grow_factor = config.gen_micro_batch // config.micro_batch_size
            assert (
                config.gen_prompt_length * grow_factor <= config.data.max_length
            ), "Error: batch is too small for specified partition"

            inputs["input_ids"] = inputs["input_ids"][:, : config.gen_prompt_length * grow_factor].reshape(
                inputs["input_ids"].size(0) * grow_factor, config.gen_prompt_length
            )

            generated_tokens, hidden_states = self.generate(
                self,
                inputs,
                config.data.max_length,
                config.gen_seq_length,
                do_sample=True,
                use_cache=True,
                include_embeds=True,
            )

            generated_tokens = generated_tokens[:, -config.gen_seq_length :]
            hidden_states = hidden_states[:, -config.gen_seq_length : -self.model.speculator.n_predict]

        preds = self.model.speculator(
            hidden_states.detach(),
            generated_tokens[:, :-1].detach(),
        )
        losses = []
        loss_fn = CrossEntropyLoss()

        labels = generated_tokens
        for i in range(preds.size(0)):
            # + 2 maps to the first speculative token
            # + 1 maps to the output token of the model
            label = labels[:, i + 1 : preds.size(2) + i + 1]  # b n
            loss = loss_fn(preds[i].reshape(-1, preds.size(3)), label.long().reshape(-1))
            losses.append(loss)

        loss = sum(losses)
        return loss

    def _compute_loss3(self, inputs):
        """
        Compute training loss using just the speculator. This loss function uses inputs to speculator directly.

        Args:
            inputs (dict): The labels, and input hidden states to speculator

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss.
        """

        inputs = to_device(inputs, self.device)
        assert "speculator_input" in inputs.keys(), "Error: speculator_input not in inputs"
        assert "speculator_label" in inputs.keys(), "Error: speculator_label not in inputs"

        preds = self.model.speculator(
            inputs["speculator_input"].detach(),
            inputs["speculator_label"].detach(),
        )
        losses = []
        loss_fn = CrossEntropyLoss()

        labels = inputs["speculator_label"]
        for i in range(preds.size(0)):
            # + 2 maps to the first speculative token
            # + 1 maps to the output token of the model
            label = labels[:, i + 1 : preds.size(2) + i + 1]  # b n
            loss = loss_fn(preds[i].reshape(-1, preds.size(3)), label.long().reshape(-1))
            losses.append(loss)

        loss = sum(losses)
        return loss

    def loss(self, batch) -> float:
        if self.config.gen_train and not self.config.gen_train_simple:
            return self._compute_loss3(batch)
        elif self.config.gen_train:
            return self._compute_loss2(batch)
        else:
            return self._compute_loss1(batch)

    def step(self, batch) -> None:
        if not (self.config.gen_train or self.config.gen_train_simple):
            super().step(batch)

        grow_factor = self.config.gen_micro_batch_size // self.config.micro_batch_size
        assert (
            self.config.gen_prompt_length * grow_factor <= self.config.data.max_length
        ), "Batch is to small for specified partition"

        inputs = to_device(batch, self.device)
        with torch.no_grad():
            inputs["input_ids"] = inputs["input_ids"][:, : self.config.gen_prompt_length * grow_factor].reshape(
                inputs["input_ids"].size(0) * grow_factor, self.config.gen_prompt_length
            )

            generated_tokens, hidden_states = self.generate(
                inputs,
                self.config.data.max_length,
                self.config.gen_seq_length,
                do_sample=True,
                use_cache=True,
                include_embeds=True,
            )

            generated_tokens = generated_tokens[:, -self.config.gen_seq_length :].reshape(
                [-1, self.config.gen_train_micro_batch_size, self.config.gen_seq_length]
            )
            hidden_states = hidden_states[:, -self.config.gen_seq_length : -self.model.speculator.n_predict, :]

            hidden_states = hidden_states.reshape(
                [
                    -1,
                    self.config.gen_train_micro_batch_size,
                    hidden_states.size(1),
                    hidden_states.size(2),
                ]
            )
        # The generation takes a long time so doing this once in a while wont be a big overhead
        torch.cuda.empty_cache()

        speculator_inputs = {}
        for i in tqdm.tqdm(
            range(generated_tokens.size(0)),
            total=generated_tokens.size(0),
            dynamic_ncols=True,
            file=sys.stdout,
            desc="Multi steps per generation: ",
            disable=torch.distributed.get_rank() != 0,
        ):
            speculator_inputs["speculator_input"] = hidden_states[i]

            # the labels are used as input to the speculator, but the last token is not used
            # since there is no further prediction to be made
            # how ever it is needed in the loss function so we leave it in tact
            speculator_inputs["speculator_label"] = generated_tokens[i]
            super().step(speculator_inputs)
