
"""

The core autograd function sequence_tiled_compute lives in Deepspeed, here we have applied versions that use it.

"""

import math
from arctic_training.deepspeed import sequence_tiled_compute
from transformers import AutoConfig
import torch

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


class TiledLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss_fn, logits, vocab_size, shift_labels, shards) -> torch.Tensor:
        """

        This is a memory efficient loss autograd function that takes the existing logits and performs loss calculation in shards.

        This one is an SFT-aware version, therefore it takes care of special cases where the whole shard is made of -100 labels and which requires then a special care.

        Note: logits seqlen dimension doesn't have to be divisible by shards, the last shard will be shorter than the rest. The calculating of the number of shards is in the example.

        Here is an example of using it:

        def loss(self, batch) -> torch.Tensor:
            batch = to_device(batch, self.device)
            shift_labels = batch.pop("shift_labels")
            outputs = self.model(**batch, use_cache=False)
            logits = outputs.logits

            if all((shift_labels == -100).squeeze()):
                # this is the case where all labels in a micro-batch are -100 (very common for SFT if the seqlen is short) - CE returns `nan` in this case, so we don't want to call loss and instead create a differentiable loss `0` which will also set all the grads to `0` in `backward` - the effect of this is akin to a perfect score where the model needs no adjustment since grads will be all zeros.
                loss = (logits.sum() * 0.0).float()

            num_shards: Any = "auto"
            if num_shards == "auto":
                # parameterize to about 1GB fp32 logits shards
                slice_size_in_gb = 1  # XXX: make configurable?
                size_in_gb = logits.numel() * 4 / 2**30  # fp32
                # the sp shard's seqlen sp shard can be easily not divisible by the derived number of chunked loss shards, so we use the uppper ceiling and allow the last chunk to be shorter than the rest
                num_shards = math.ceil(size_in_gb / slice_size_in_gb)
                # print(f"derived {num_shards} shards for size {size_in_gb}GB")
            if num_shards > 1:
                # if shards == 1 this will lead to a higher memory usage then calling the normal loss function, so don't do that.
                loss = TiledLoss.apply(
                    self.model_unwrapped.loss_function,
                    logits,
                    self.model_unwrapped.config.vocab_size,
                    shift_labels,
                    num_shards,
                )
            else:
                loss = self.model_unwrapped.loss_function(
                    logits=logits,
                    labels=None,
                    vocab_size=self.model_unwrapped.config.vocab_size,
                    shift_labels=shift_labels,
                )

            return loss


        """
        ctx.save_for_backward(logits, shift_labels)
        ctx.loss_fn = loss_fn
        ctx.vocab_size = vocab_size
        ctx.shards = shards

        with torch.no_grad():
            seqlen = shift_labels.shape[1]
            shard_step = math.ceil(seqlen / shards)
            loss_shards = []
            total_good_items = 0

            # since -100s are ignored we have to perform a weighted average on each loss slice as each slice may contribute a different number of non- -100 labels
            # if seqlen / shards != 0 - the last chunk is just shorter than the rest but no data is ignored
            for i in range(shards):
                # XXX: here and everywhere don't make a copy, pass the slice or perhaps narrow/view?
                shift_labels_shard = shift_labels[:, i * shard_step : (i + 1) * shard_step]
                if all((shift_labels_shard == -100).squeeze()):
                    continue  # ignore this shard
                loss_shard = loss_fn(
                    logits=logits[:, i * shard_step : (i + 1) * shard_step, :],
                    labels=None,
                    vocab_size=vocab_size,
                    shift_labels=shift_labels_shard,
                )
                good_items = sum((shift_labels_shard != -100).squeeze())
                loss_shards.append(loss_shard * good_items)
                total_good_items += good_items
            total_loss = torch.cat([l.unsqueeze(0) for l in loss_shards], dim=0).sum()
            weighted_loss = total_loss / total_good_items

        # weighted_loss.requires_grad = True
        return weighted_loss

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:

        logits, shift_labels = ctx.saved_tensors
        loss_fn = ctx.loss_fn
        vocab_size = ctx.vocab_size
        shards = ctx.shards

        grad = grads[0]
        logits_grad = torch.zeros_like(logits)
        # logits_grad = torch.zeros(logits.shape, device=logits.device, dtype=grad.dtype, requires_grad=logits.requires_grad)

        logits_shards = list(torch.chunk(logits, chunks=shards, dim=1))
        shift_labels_shards = list(torch.chunk(shift_labels, chunks=shards, dim=1))
        del logits
        del shift_labels
        ctx.logits = None
        ctx.shift_labels = None
        ctx.loss_fn = None
        ctx.vocab_size = None
        ctx.shards = None

        # if seqlen is not exactly divisible by shards the last step will be shorter than shard_step
        shard_step = logits_shards[0].numel()
        for i in range(shards):
            logits_shard = logits_shards.pop(0)
            shift_labels_shard = shift_labels_shards.pop(0)

            shard_offset = i * shard_step
            # this will enable gradual population of the pre-allocated `logits_shard.grad` during `torch.autograd.backward` calls
            logits_shard.grad = (
                logits_grad.view(-1).narrow(0, shard_offset, logits_shard.numel()).view_as(logits_shard)
            )

            with torch.enable_grad():
                if all((shift_labels_shard == -100).squeeze()):
                    # fake loss calculation, since CE will return nan, but grads will be set
                    # a normal loss_fn upcasts logits to float so match it
                    loss_shard = (logits_shard.sum() * 0.0).float()
                else:
                    loss_shard = loss_fn(
                        logits=logits_shard.requires_grad_(),
                        labels=None,
                        vocab_size=vocab_size,
                        shift_labels=shift_labels_shard,
                    )

            torch.autograd.backward(loss_shard, grad)

        logits_grad /= shards

        # print(f"returning {logits_grad.norm()=}")
        # print(f"returning {logits_grad=}")
        # only logits (2nd arg) needs grads
        return None, logits_grad, None, None, None

