from typing import Dict
from typing import List


def get_optimizer_grouped_parameters(
    model,
    weight_decay: float,
    no_decay_name_list: List[str] = [
        "bias",
        "layer_norm.weight",
        "layernorm.weight",
        "norm.weight",
        "ln_f.weight",
    ],
) -> List[Dict]:
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                )
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                )
            ],
            "weight_decay": 0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups
