from pathlib import Path
from typing import Any

import deepspeed
import torch
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from arctic_training.checkpoint import CheckpointEngine
from arctic_training.register import register_checkpoint


@register_checkpoint
class HFCheckpointEngine(CheckpointEngine):
    checkpoint_type = "huggingface"

    @property
    def model_file(self) -> Path:
        return self.checkpoint_dir / "pytorch_model.bin"

    @property
    def config_file(self) -> Path:
        return self.checkpoint_dir / "config.json"

    @staticmethod
    def _get_param(param: Any) -> Any:
        if hasattr(param, "ds_id"):
            params_to_fetch = []
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                params_to_fetch = [param]
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=True):
                return param.data.cpu()
        else:
            return param.cpu()

    def _save_z3_checkpoint(self) -> None:
        output_state_dict = {}
        for k, v in self.model.named_parameters():
            v_p = self._get_param(v)
            if self.trainer.global_rank == 0:
                output_state_dict[k] = v_p

        if self.trainer.global_rank == 0:
            torch.save(output_state_dict, self.model_file)

        del output_state_dict

    def save(self) -> None:
        if self.trainer.config.zero_3_enabled:  # TODO implement this
            self._save_z3_checkpoint()
        else:
            torch.save(self.model, self.model_file)
            self.model.config.to_json_file(self.config_file)
            self.trainer.tokenizer.save_pretrained(
                self.checkpoint_dir
            )  # TODO save tokenizer in trainer

    def load(self) -> None:
        raise NotImplementedError
