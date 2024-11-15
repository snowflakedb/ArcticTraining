from typing import TYPE_CHECKING
from typing import Optional

from peft import get_peft_model
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import PreTrainedModel

if TYPE_CHECKING:
    from arctic_training.config import ModelConfig
    from arctic_training.trainer import Trainer


def hf_model_loader(model_config: "ModelConfig") -> PreTrainedModel:
    hf_model_config = AutoConfig.from_pretrained(model_config.name_or_path)

    if model_config.use_liger_kernel:
        try:
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
        except ImportError:
            raise ImportError(
                "You need to install the liger-kernel package to use LigerKernel models: `pip install liger-kernel`"
            )
        model = AutoLigerKernelForCausalLM.from_pretrained(
            model_config.name_or_path,
            config=hf_model_config,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=model_config.dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.name_or_path,
            config=hf_model_config,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=model_config.dtype,
        )

    return model


def model_factory(
    trainer: "Trainer", model_config: Optional["ModelConfig"] = None
) -> PreTrainedModel:
    if model_config is None:
        model_config = trainer.config.model

    model_load_fn = hf_model_loader
    if trainer.model_loader is not None:
        model_load_fn = trainer.model_loader

    model = model_load_fn(model_config)

    for adjustment in model_config.adjustments:
        model = adjustment(model)

    if model_config.peft_config:
        model = get_peft_model(model, model_config.peft_config)

    if not model_config.disable_activation_checkpoint:
        model.gradient_checkpointing_enable()
        model = make_model_gradient_checkpointing_compatible(model)

    return model


def make_model_gradient_checkpointing_compatible(
    model: PreTrainedModel,
) -> PreTrainedModel:
    # Higgingface added this enable input require grads function to make gradient checkpointing work for lora-only optimization
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model
