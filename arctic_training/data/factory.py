from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase

from arctic_training.data.loader import ConcatenatedDataSetsLoader
from arctic_training.logging import logger

if TYPE_CHECKING:
    from arctic_training.config import DataConfig
    from arctic_training.trainer import Trainer
IGNORE_INDEX = -100


# this function is modified from TRL trl.trainer.utils.py
def pad(
    tensors: List[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    is_position_id: bool = False,
    divisible_by: int = 256,
    max_seq: Optional[int] = None,
    dim_to_pad: int = -1,
) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.

    Args:
        tensors (`List[torch.Tensor]`):
            List of input tensors to pad.
        padding_value (`int`):
            Value to use for padding. Default is 0.
        padding_side (`str`):
            Side on which to add padding. Must be 'left' or 'right'. Default is 'right'.
        is_position_id (`bool`):
            If it is position_id, we will use arange to generate the position id in order to avoid too much padding causes flash attn crash.
        divisible_by (`int`):
            The number that the length of the sequence should be divisible by.
        max_seq (`int`):
            The maximum length of the sequence. If it is not None, we will truncate the sequence to the maximum length or pad the sequence to the maximum length.
        dim_to_pad (`int`):
            The dimension to pad. Default is -1.
    Returns:
        `torch.Tensor`:
            A single tensor containing the padded tensors.

    Examples:
        >>> import torch
        >>> pad([torch.tensor([1, 2, 3]), torch.tensor([4, 5])])
        tensor([[1, 2, 3],
                [4, 5, 0]])
        >>> pad([torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])])
        tensor([[[1, 2],
                [3, 4]],

                [[5, 6],
                [0, 0]]])
    """
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()
    if max_seq is not None:
        output_shape[dim_to_pad] = max_seq
    elif divisible_by is not None:
        output_shape[dim_to_pad] = (
            int(np.ceil(output_shape[dim_to_pad] / divisible_by)) * divisible_by
        )

    # Create an output tensor filled with the padding value
    # TODO: Likely for 2D position ids, this does not work. Need to revisit.
    if is_position_id:
        output = (
            torch.arange(
                output_shape[dim_to_pad],
                dtype=tensors[0].dtype,
                device=tensors[0].device,
            )
            .repeat(len(tensors) * np.prod(output_shape) // output_shape[dim_to_pad])
            .view(len(tensors), *output_shape)
        )
    else:
        output = torch.full(
            (len(tensors), *output_shape),
            padding_value,
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")
        # import pdb; pdb.set_trace()
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t
    return output


class DataCollatorForCausalLM:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(example["input_ids"]) for example in instances]
        labels = [torch.tensor(example["labels"]) for example in instances]
        # https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_flash_attention_utils.py#L270
        # we do not need attention_mask when pos-id is provided and multi-seq packed
        # attention_mask = [
        #     torch.tensor(example["attention_mask"]) for example in instances
        # ]
        if "position_ids" in instances[0]:
            position_ids = [
                torch.tensor(example["position_ids"]) for example in instances
            ]
        else:
            position_ids = [
                torch.tensor(list(range(len(example["input_ids"]))))
                for example in instances
            ]

        input_ids = pad(input_ids, padding_value=self.tokenizer.pad_token_id)
        labels = pad(labels, padding_value=IGNORE_INDEX)
        position_ids = pad(position_ids, padding_value=0, is_position_id=True)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }


def data_factory(
    trainer: "Trainer", data_config: Optional["DataConfig"] = None
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if data_config is None:
        data_config = trainer.config.data

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        data_config.tokenizer_name_or_path
    )

    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(
            f"Tokenizer {data_config.tokenizer_name_or_path} does not have pad token, we set it to eos token!"
        )

    train_dataset_loader = ConcatenatedDataSetsLoader(
        dataset_list=data_config.datasets,
        tokenizer=tokenizer,
        eval=False,
        config=data_config,
    )
    train_dataset = train_dataset_loader.load_datasets()
    eval_dataset = None

    if data_config.eval_datasets:
        eval_dataset_loader = ConcatenatedDataSetsLoader(
            dataset_list=data_config.eval_datasets,
            tokenizer=tokenizer,
            eval=True,
            config=data_config,
        )
        eval_dataset = eval_dataset_loader.load_datasets()
    elif data_config.train_eval_split[1] > 0.0:
        tmp = train_dataset.train_test_split(
            test_size=data_config.train_eval_split[1], seed=data_config.seed
        )
        train_dataset = tmp["train"]
        eval_dataset = tmp["test"]
        del tmp

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=DataCollatorForCausalLM(tokenizer=tokenizer),
        batch_size=trainer.config.micro_batch_size,
        sampler=RandomSampler(train_dataset),
        num_workers=data_config.num_proc,
        drop_last=True,
    )
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=DataCollatorForCausalLM(tokenizer=tokenizer),
            batch_size=trainer.config.micro_batch_size,
            sampler=RandomSampler(eval_dataset),
            num_workers=data_config.num_proc,
            drop_last=True,
        )

    return tokenizer, train_dataloader, eval_dataloader
