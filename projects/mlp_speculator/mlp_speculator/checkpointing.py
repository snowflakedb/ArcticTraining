import copy
import os

import torch
import torch.distributed as dist
from deepspeed.runtime.zero import GatheredParameters

from arctic_training.checkpoint import CheckpointEngine
from arctic_training.register import register_checkpoint

from .speculator import MLPSpeculator


@register_checkpoint
class MLPSpeculatorCheckpointEngine(CheckpointEngine):
    checkpoint_type: str = "mlp_speculator"

    def load(self) -> None:
        raise NotImplementedError()

    def save(self) -> None:
        if dist.get_rank() == 0:
            model_config = copy.deepcopy(self.model.speculator.config)
            model_to_save = MLPSpeculator(model_config)
            parameters_to_save = model_to_save.parameters()
        else:
            parameters_to_save = [None for param in self.model.speculator.parameters()]

        dist.barrier()

        # Gather final model.
        for parameter_to_save, (name, ds_param) in zip(
            parameters_to_save, self.model.speculator.named_parameters()
        ):
            with GatheredParameters(ds_param):
                if dist.get_rank() == 0:
                    parameter_to_save.data.copy_(ds_param.data)

        if dist.get_rank() == 0 and self.config.output_dir is not None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            save_path = os.path.join(self.config.output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), save_path)
            model_config.save(self.config.output_dir)

        dist.barrier()
