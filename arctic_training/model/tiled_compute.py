import math
from arctic_training.deepspeed import sequence_tiled_compute
from transformers import AutoConfig

def get_model_type(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path)
    return config.model_type

def tiled_mlp_forward_llama(self, x):
    """  a monkey patch to replace modeling_llama.LlamaMLP.forward to performed a tiled compute of the same """
    # import os
    # rank = int(os.getenv("LOCAL_RANK", 0))
    # if rank == 0:
    #     print(f"computing main {x.shape}")

    # XXX: temp
    #num_shards = 16
    num_shards = "auto"

    if num_shards == "auto":
        bs, seqlen, hidden = x.shape
        # XXX: not too many?
        num_shards = math.ceil(seqlen/hidden)
        #print(f"derived {num_shards} for {seqlen=} and {hidden=}")

    kwargs_to_shard = dict(x=x)
    kwargs_to_pass = dict(self=self)
    grad_requiring_tensor_key = "x"
    compute_params = [self.down_proj.weight, self.gate_proj.weight, self.up_proj.weight]
    seqlen = x.shape[1]

    def mlp_forward(self=None, x=None):
        #print(f"computing sub {x.shape}")
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    x = sequence_tiled_compute(
        mlp_forward,
        seqlen,
        num_shards,
        kwargs_to_shard,
        kwargs_to_pass,
        grad_requiring_tensor_key,
        compute_params,
        output_unshard_dimension=1, # x
        output_reduction=None,
    )
    return x


def enable_tiled_mlp_compute(model_name_or_path):
    """
    Important: this monkey patching call, that overrides the original HF Transformers model's MLP class, has to happen before model is instantiated.
    Currently only some models are supported, but we can easily add support for more model architectures if needed.
    """

    model_type = get_model_type(model_name_or_path)
    if model_type == "llama":
        from transformers.models.llama import modeling_llama
        modeling_llama.LlamaMLP.forward = tiled_mlp_forward_llama
    else:
        raise ValueError(f"model type {model_type} is currently not supported. Please open an issue and ask to add Tiled MLP support for {model_type}.")
