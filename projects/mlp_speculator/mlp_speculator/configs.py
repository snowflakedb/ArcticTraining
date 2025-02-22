import json
import os

import transformers

from arctic_training.config import Config


# Configs used for training the model
class MLPSpeculatorTrainConfig(Config):
    speculator_width: str = "3072"
    n_speculator_heads: int = 3
    speculator_tie_weights: bool = False
    speculator_scale_input: bool = False
    speculator_path: str = "None"
    gen_train: bool = False
    gen_train_simple: bool = False
    gen_micro_batch: int = 32
    gen_train_micro_batch: int = 32
    gen_prompt_length: int = 64
    sim_gen_loss: bool = False
    loss_type: str = ""
    ctc_loss_weight: float = 0.0
    gen_seq_length: int = 256
    freeze_layers: list = []
    weighted_sum: bool = False
    param_init_method: str = "zeros"


# Configs used for savings model checkpoint for inference
class MLPSpeculatorConfig:
    """
    This is a simple MLP-based speculator Config.
    ...
    Args
    ----
    emb_dim : int
        Dimensionality of the input vector from the base model.
    inner_dim : str
        Latent dimensionality of the speculator model.
    vocab_size : int
        Number of entries in the tokenizer associated with the base model.
    n_predict : int
        Number of heads / number of tokens to guess ahead. Model size and speed scale with this value.
    tie_weights : bool
        If true, use a single set of weights for every model head/stage after the first.
        The initial projection from the base model may have a different size, so that stays separate.
    scale_input: bool
        If true, apply an extra layernorm to the initial state vector input.
        Helps training dynamics, particularly when base model output has unusual scale.
    """

    def __init__(
        self,
        base_model_name_or_path,
        emb_dim,
        inner_dim,
        vocab_size,
        n_predict,
        tie_weights=False,
        scale_input=False,
    ):
        self.architectures = ["MLPSpeculatorPreTrainedModel"]
        self.base_model_name_or_path = base_model_name_or_path

        self.emb_dim = emb_dim
        self.inner_dim = inner_dim
        self.model_type = "mlp_speculator"

        self.n_candidates = n_predict
        self.n_predict = n_predict

        self.scale_input = scale_input
        self.tie_weights = tie_weights
        self.top_k_tokens_per_head = [1 for i in range(self.n_predict)]

        self.torch_dtype = "bfloat16"
        self.transformers_version = transformers.__version__
        self.vocab_size = vocab_size

    def save(self, output_dir):
        save_path = os.path.join(output_dir, "config.json")
        with open(save_path, "w") as f:
            json.dump(self.__dict__, f)
