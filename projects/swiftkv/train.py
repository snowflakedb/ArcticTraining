import tempfile
tempfile.tempdir = '/data-fast/tmp'

import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers.modeling_utils import no_init_weights

import copy
from typing import Any
from typing import TYPE_CHECKING

from deepspeed.runtime.zero import GatheredParameters

from arctic_training.trainer.sft_trainer import SFTTrainer
from llama_swiftkv import LlamaSwiftKVConfig
from llama_swiftkv import LlamaSwiftKVForCausalLM

from arctic_training.config import ModelConfig

from arctic_training.config import Config, DataConfig
from arctic_training.trainer.sft_trainer import to_device
from arctic_training.checkpoint import CheckpointEngine


class SwiftKVConfig(Config):
    num_key_value_layers: int = None
    key_value_group_size: int = 1
    decoder_loss_mult : float = 0.0
    temperature: float = 1.0
    zero: int = 2
    model_path: str = None


def init_student_layers(self):
    # Freeze all teacher parameters
    for param in self.model.parameters():
        param.requires_grad = False

    # Initialize student layers
    self.model.model.norm_swiftkv.weight.requires_grad = True
    for layer in self.model.model.layers[self.hf_model_config.num_key_value_layers :]:
        # Initialize q_proj_swiftkv
        with GatheredParameters(layer.parameters(), modifier_rank=0):
            layer.self_attn.q_proj_swiftkv.weight.data.copy_(
                layer.self_attn.q_proj.weight.data
            )
        layer.self_attn.q_proj_swiftkv.weight.requires_grad = True
    for layer_idx in range(
        self.hf_model_config.num_key_value_layers,
        self.hf_model_config.num_hidden_layers,
        self.hf_model_config.key_value_group_size,
    ):
        layer = self.model.model.layers[layer_idx]
        # Initialize k_proj_swiftkv
        k_proj_weights = [layer.self_attn.k_proj_swiftkv.weight] + [
            self.model.model.layers[layer_idx + i].self_attn.k_proj.weight
            for i in range(self.config.key_value_group_size)
        ]
        with GatheredParameters(k_proj_weights, modifier_rank=0):
            k_proj_weights[0].data.copy_(
                sum(k_proj_weights[1:]) / self.config.key_value_group_size
            )
        layer.self_attn.k_proj_swiftkv.weight.requires_grad = True
        # Initialize v_proj_swiftkv
        v_proj_weights = [layer.self_attn.v_proj_swiftkv.weight] + [
            self.model.model.layers[layer_idx + i].self_attn.v_proj.weight
            for i in range(self.config.key_value_group_size)
        ]
        with GatheredParameters(v_proj_weights, modifier_rank=0):
            v_proj_weights[0].data.copy_(
                sum(v_proj_weights[1:]) / self.config.key_value_group_size
            )
        layer.self_attn.v_proj_swiftkv.weight.requires_grad = True
    self.model.gradient_checkpointing_enable()


class SwiftKVCheckpointEngine(CheckpointEngine):
    checkpoint_type: str = "swiftkv"

    def load(self) -> None:
        raise NotImplementedError()

    def save(self) -> None:
        if dist.get_rank() == 0:
            model_config = copy.deepcopy(self.model.module.config)
            with no_init_weights():
                final_model = LlamaSwiftKVForCausalLM(model_config).swiftkv(True)
            final_parameters = final_model.parameters()
        else:
            final_parameters = [None for _ in self.model.parameters()]

        dist.barrier()

        # Gather final model.
        for hf_param, (name, ds_param) in zip(
            final_parameters, self.model.named_parameters()
        ):
            with GatheredParameters(ds_param):
                if dist.get_rank() == 0:
                    hf_param.data.copy_(ds_param.data)

        if dist.get_rank() == 0 and self.config.output_dir is not None:
            final_model.save_pretrained(
                self.config.output_dir,
                safe_serialization=True,
                max_shard_size="4GB",
            )
            self.trainer.tokenizer.save_pretrained(self.config.output_dir)

        dist.barrier()


class SwiftKVTrainer(SFTTrainer):
    config_type = SwiftKVConfig
    _trainer_callbacks = [("post-model-init", init_student_layers)]

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.model_config = None
        self.epoch = 0

    def checkpoint_engine(self):
        ckpt_engine = SwiftKVCheckpointEngine(trainer=self, config=self.config.checkpoint[0])
        return [ckpt_engine]

    def model_loader(self, model_config: "ModelConfig") -> Any:
        #TODO(jeff): change model_path to model_name_or_path
        hf_model_config = LlamaSwiftKVConfig.from_pretrained(self.config.model_path)
        hf_model_config.num_key_value_layers = self.config.num_key_value_layers
        hf_model_config.key_value_group_size = self.config.key_value_group_size
        self.hf_model_config = hf_model_config
        model = LlamaSwiftKVForCausalLM.from_pretrained(
            self.config.model_path,
            config=copy.deepcopy(hf_model_config),
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        return model

    def loss(self, batch):
        batch = to_device(batch, self.device)

        with torch.no_grad():
            self.model.swiftkv(False)
            self.model.eval()
            teacher_outputs = self.model(
                **batch,
                output_hidden_states=(self.config.decoder_loss_mult > 0),
            )

        self.model.swiftkv(True)
        self.model.train()
        student_outputs = self.model(
            **batch,
            output_hidden_states=(self.config.decoder_loss_mult > 0),
        )

        distill_loss = self.distillation_loss(
            student_outputs.logits,
            teacher_outputs.logits,
            temperature=self.config.temperature,
        )

        decoder_loss = torch.zeros_like(distill_loss)
        if self.config.decoder_loss_mult > 0:
            decoder_loss_count = 0
            for layer_idx in [15, 23]:
                student_hidden = student_outputs.hidden_states[layer_idx]
                teacher_hidden = teacher_outputs.hidden_states[layer_idx]
                decoder_loss += torch.linalg.norm(
                    student_hidden - teacher_hidden,
                    dim=-1,
                ).mean()
                decoder_loss_count += 1
            decoder_loss *= self.config.decoder_loss_mult / decoder_loss_count

        if dist.get_rank() == 0:
            print(
                "student loss:",
                student_outputs.loss.item(),
                "teacher loss:",
                teacher_outputs.loss.item(),
                "distill loss:",
                distill_loss.item(),
                "decoder loss:",
                decoder_loss.item(),
            )

        loss = distill_loss + decoder_loss

        torch.cuda.synchronize()

        return loss


    def distillation_loss(self, student_output, teacher_output, temperature=1.0, dim=-1):
        # Soften the student logits by applying softmax first and log() second
        soft_targets = F.softmax(teacher_output / temperature, dim=dim)
        soft_prob = F.log_softmax(student_output / temperature, dim=dim)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the
        # authors of the paper "Distilling the knowledge in a neural network"
        return torch.mean(
            torch.sum(
                soft_targets * (soft_targets.log() - soft_prob),
                dim=dim,
            )
            * temperature**2
        )


if __name__ == "__main__":
    model_path = "/checkpoint/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/5206a32e0bd3067aef1ce90f5528ade7d866253f"

    datasets = ["HuggingFaceH4/ultrachat_200k",
                "meta-math/MetaMathQA",
                "ise-uiuc/Magicoder-OSS-Instruct-75K",
                "lmsys/lmsys-chat-1m",
                "Open-Orca/SlimOrca"]

    data_config = DataConfig(
        tokenizer=model_path,
        datasets=datasets,
        use_data_cache=True,
        cache_processed_data=True,
        data_cache_dir="/data-fast/st-data-new",
        num_proc=16,
        max_length=8192,
    )

    model_config = ModelConfig(
        name_or_path=model_path,
        use_liger_kernel=False,
        disable_activation_checkpoint=True,
    )

    output_dir = "/checkpoint/swiftkv/llama-swiftkv-8b-oss-ultra-math-magic-lmsys-orca-r2"

    config = SwiftKVConfig(
        num_key_value_layers=16,
        key_value_group_size=1,
        lr=0.0002,
        warmup_ratio=0.05,
        deepspeed={"zero_optimization": {
        "stage": 2, 
        "stage3_param_persistence_threshold": 1.000000e+04, 
        "stage3_max_live_parameters": 3.000000e+07, 
        "stage3_prefetch_bucket_size": 3.000000e+07, 
        "memory_efficient_linear": False
    }}, 
        decoder_loss_mult=0.0,
        gradient_accumulation_steps=1,
        betas=(0.9, 0.999),
        seed=42,
        # ckpt_save_interval=1000,
        # eval_frequency=0,
        epochs=1,
        micro_batch_size=1,
        zero=2,
        weight_decay=0.0,
        temperature=2.0,
        data=data_config,
        model=model_config,
        model_path=model_path,
        checkpoint={"type":"huggingface", "output_dir": output_dir, "save_every_n_steps":1000, "save_every_n_epochs":1},
    )

    trainer = SwiftKVTrainer(config)
    trainer.train()
