# General Imports
import os
import sys
import types
from typing import Optional

import torch
import torch.nn.functional as F
import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn import CTCLoss
from transformers.cache_utils import DynamicCache
from transformers.trainer_pt_utils import LabelSmoother

from arctic_training.logging import logger

# ArcticTraining Imports
from arctic_training.trainer.sft_trainer import SFTTrainer
from arctic_training.trainer.sft_trainer import to_device

from .configs import MLPSpeculatorConfig

# MLPSpeculator Imports
from .speculator import MLPSpeculator

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
# logger = LOG = logging.getLogger("mlp_speculator")


class MLPSpeculatorTrainer(SFTTrainer):
    def add_mlp_speculator(self):
        """
        Args:
            self (nn.Module): self is the trainer.
        """
        model = self.model
        config = self.config

        hidden_size = model.lm_head.in_features
        vocab_size = model.lm_head.out_features
        model.config.n_speculator_heads = config.n_speculator_heads
        model.n_speculator_heads = config.n_speculator_heads

        speculator_config = MLPSpeculatorConfig(
            config.model.name_or_path,
            hidden_size,
            config.speculator_width,
            vocab_size,
            config.n_speculator_heads,
            tie_weights=config.speculator_tie_weights,
            scale_input=config.speculator_scale_input,
        )

        model.speculator = MLPSpeculator(speculator_config)

        # Ensure Speculator dtype and device align with the base_model
        model.speculator.to(model.dtype).to(model.device)

        model.speculator.reset_parameters()

        # if config.speculator_path is not None:
        #     model_state_dict = torch.load(config.speculator_path)

        #     model.speculator.load_state_dict(model_state_dict)

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

    def set_trainable_parameters_for_speculator(self):
        """
        Args:
            self (nn.Module): self is the trainer.
        """
        model = self.model
        if model.n_speculator_heads is not None:
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze speculator heads
            for param in model.speculator.parameters():
                param.requires_grad = True

    _trainer_callbacks = [
        ("post-model-init", add_mlp_speculator),
        ("post-model-init", set_trainable_parameters_for_speculator),
    ]

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
        assert (
            type(input_ids) is torch.Tensor and input_ids.dim() == 2
        ), "Invalid Input Shape. Must be b x n"

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

    def _compute_loss_simulate_generation(self, inputs):
        """
        We saved the input and generated tokens by compute_loss3.
        """
        inputs = to_device(inputs, self.device)
        labels = inputs.pop("labels")

        with torch.no_grad():
            outputs = self.model(**inputs, speculator_return=True)
            hidden_states = outputs[0]  # b n h

        gen_seq_length = labels.shape[-1]

        spec_inputs = hidden_states.detach()[
            :, -gen_seq_length - 1 : -self.model.speculator.n_predict - 1, :
        ]
        spec_inputs2 = labels[:, -gen_seq_length:]
        preds = self.model.speculator(spec_inputs, spec_inputs2)
        losses = []
        loss_fn = CrossEntropyLoss()

        for i in range(preds.size(0)):
            label = labels[:, i + 1 : preds.size(2) + i + 1]  # b n
            loss = loss_fn(
                preds[i].reshape(-1, preds.size(3)), label.long().reshape(-1)
            )
            losses.append(loss)
        loss = sum(losses)
        return loss

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
        weighted_sum = self.config.weighted_sum
        aurick_loss = self.config.aurick_loss
        ctc_loss_weight = self.config.ctc_loss_weight

        if aurick_loss:
            total_loss = 0
            for i in range(preds.size(0)):
                targ = labels[:, i + 2 : preds.size(2) + i + 2]  # b n
                loss = torch.sum(
                    torch.softmax(preds[i].reshape(-1, preds.size(3)), dim=-1)
                    * F.one_hot(targ.long().reshape(-1), preds.size(3)),
                    dim=-1,
                )
                losses.append(loss)
                cur_loss = loss
                for j in range(i):
                    cur_loss = cur_loss * losses[j]
                total_loss += cur_loss
            loss = -torch.sum(torch.log(total_loss + 1e-7))
        elif ctc_loss_weight:
            ctc_loss_fn = CTCLoss(blank=128002, zero_infinity=True)
            targets = []
            for i in range(preds.size(0)):
                targ = labels[:, i + 2 : preds.size(2) + i + 2]  # b n
                targets.append(targ.reshape(-1))
                loss = loss_fn(
                    preds[i].reshape(-1, preds.size(3)), targ.long().reshape(-1)
                )
                losses.append(loss)
            loss = sum(losses)
            targets = torch.stack(targets, dim=1)
            logsoftmaxed = (
                preds.view(preds.shape[0], -1, preds.shape[-1]).float().log_softmax(2)
            )
            lengths = torch.full(
                (targets.shape[0],),
                preds.shape[0],
                dtype=torch.long,
                device=preds.device,
            )
            loss += ctc_loss_weight * ctc_loss_fn(
                logsoftmaxed, targets, lengths, lengths
            )
        else:
            for i in range(preds.size(0)):
                targ = labels[:, i + 2 : preds.size(2) + i + 2]  # b n
                loss = loss_fn(
                    preds[i].reshape(-1, preds.size(3)), targ.long().reshape(-1)
                )
                if weighted_sum:
                    weight = 1.0 / float(1.0 + i)
                    losses.append(weight * loss)
                else:
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

            inputs["input_ids"] = inputs["input_ids"][
                :, : config.gen_prompt_length * grow_factor
            ].reshape(
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
            hidden_states = hidden_states[
                :, -config.gen_seq_length : -self.model.speculator.n_predict
            ]

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
            loss = loss_fn(
                preds[i].reshape(-1, preds.size(3)), label.long().reshape(-1)
            )
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
        assert (
            "speculator_input" in inputs.keys()
        ), "Error: speculator_input not in inputs"
        assert (
            "speculator_label" in inputs.keys()
        ), "Error: speculator_label not in inputs"

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
            loss = loss_fn(
                preds[i].reshape(-1, preds.size(3)), label.long().reshape(-1)
            )
            losses.append(loss)

        loss = sum(losses)
        return loss

    def loss(self, inputs):
        config = self.config

        # train on generated data distribution
        # does generation inside the loss function
        if config.sim_gen_loss:
            return self._compute_loss_simulate_generation(inputs)

        if config.gen_train and not config.gen_train_simple:
            assert (
                config.gen_micro_batch % config.gen_train_micro_batch == 0
            ), "Error: gen_micro_batch must be divisible by gen_train_micro_batch"

            return self._compute_loss3(inputs)

        # train on generated data distribution
        # does generation outside the loss function
        # can generate data for multiple steps to be more efficient
        elif config.gen_train:
            assert (
                config.gen_train_micro_batch == config.gen_micro_batch
            ), "Error: gen_train_micro_batch must be equal to gen_micro_batch"
            return self._compute_loss2(inputs)

        # does not do any generation. Train on the original data distribution
        else:
            return self._compute_loss1(inputs)

    # #overriding the train_batch_loop method to include gen_train
    def train_batch_loop(self) -> None:
        self._step()
        if self.local_rank == 0:
            batch_factor = (
                1
                if not self.config.gen_train
                else self.config.gen_micro_batch / self.config.gen_train_micro_batch
            )
            logger.info(
                f"EPOCH: {self.epoch_idx}, TRAIN BATCH: {self.train_batch_idx * batch_factor}, GLOBAL_STEP: {self.global_step_idx}, LOSS: {self._loss_output.item()}"
            )

    def step(self):
        """
        Perform a single training step or single generation with multiple training steps.

        """
        config = self.config

        if not config.gen_train or config.gen_train_simple:
            super().step()
            return

        grow_factor = config.gen_micro_batch // config.micro_batch_size
        assert (
            config.gen_prompt_length * grow_factor <= config.data.max_length
        ), "Error: batch is too small for specified partition"

        parent_step = super().step

        def multi_step_with_generation():
            inputs = to_device(self.train_batch_data, self.device)
            with torch.no_grad():
                inputs["input_ids"] = inputs["input_ids"][
                    :, : config.gen_prompt_length * grow_factor
                ].reshape(
                    inputs["input_ids"].size(0) * grow_factor, config.gen_prompt_length
                )

                rank = torch.distributed.get_rank()
                os.makedirs(f"toks/{rank}", exist_ok=True)

                generated_tokens, hidden_states = self.generate(
                    inputs,
                    config.data.max_length,
                    config.gen_seq_length,
                    do_sample=True,
                    use_cache=True,
                    include_embeds=True,
                )

                generated_tokens = generated_tokens[
                    :, -config.gen_seq_length :
                ].reshape([-1, config.gen_train_micro_batch, config.gen_seq_length])
                hidden_states = hidden_states[
                    :, -config.gen_seq_length : -self.model.speculator.n_predict, :
                ]

                file_length = len(os.listdir(f"toks/{rank}"))
                torch.save(
                    {"input": inputs["input_ids"], "generated": generated_tokens},
                    f"toks/{rank}/{file_length:06d}.pt",
                )

                hidden_states = hidden_states.reshape(
                    [
                        -1,
                        config.gen_train_micro_batch,
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
                self.train_batch_data = speculator_inputs
                parent_step()

        # When this runs, we are using compute_loss3
        multi_step_with_generation()

    # def checkpoint_engine(self):

    #     ckpt_engine = MLPSpeculatorCheckpointEngine(
    #         trainer=self, config=self.config.checkpoint[0]
    #     )
    #     return [ckpt_engine]

    # WIP
    # def get_average_accepted_tokens(config, sft_trainer, inputs):
    # """
    # Compute the training loss for the model.

    # Args:
    #     inputs (dict): The input data, including input IDs, attention mask, and labels.

    # Returns:
    #     Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
    # """

    #     inputs = to_device(inputs, sft_trainer.device)

    #     with torch.no_grad():
    #         grow_factor = config.gen_micro_batch // config.micro_batch_size
    #         assert (
    #             config.gen_prompt_length * grow_factor <= config.data.max_length
    #         ), "Error: batch is too small for specified partition"

    #         inputs["input_ids"] = inputs["input_ids"][
    #             :, : config.gen_prompt_length * grow_factor
    #         ].reshape(inputs["input_ids"].size(0) * grow_factor, config.gen_prompt_length)

    #         generated_tokens, hidden_states = self.generate(
    #             inputs,
    #             config.data.max_length,
    #             config.gen_seq_length,
    #             do_sample=True,
    #             use_cache=True,
    #             include_embeds=True,
    #         )

    #         generated_tokens = generated_tokens[:, -config.gen_seq_length :]

    #         for_speculation = generated_tokens[:, : -sft_trainer.model.speculator.n_predict]

    #         hidden_states = hidden_states[
    #             :, -config.gen_seq_length : -sft_trainer.model.speculator.n_predict
    #         ]

    #     # [bxs,n]
    #     predictions = sft_trainer.model.speculator.generate_suffixes(
    #         hidden_states.reshape(-1, hidden_states.size(2)),  # [bxs,h]
    #         for_speculation.reshape(-1),  # [bxs]
    #         1,
    #         1,
    #     )

    #     for_spec_token_len = for_speculation.size(1) - 1
    #     generated_tokens_expanded = torch.concat(
    #         [
    #             generated_tokens[:, 1 : for_spec_token_len + 1].unsqueeze(0),
    #             generated_tokens[:, 2 : for_spec_token_len + 2].unsqueeze(0),
    #             generated_tokens[:, 3 : for_spec_token_len + 3].unsqueeze(0),
    #         ],
    #         dim=0,
    #     )

    #     predictions = predictions.reshape(
    #         -1, for_speculation.size(1), config.n_speculator_heads
    #     )
