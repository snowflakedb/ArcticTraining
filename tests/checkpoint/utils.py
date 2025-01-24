import torch


def models_are_equal(model_a: torch.nn.Module, model_b: torch.nn.Module) -> bool:
    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        if not param_a.data.eq(param_b.data).all():
            return False

    return True
