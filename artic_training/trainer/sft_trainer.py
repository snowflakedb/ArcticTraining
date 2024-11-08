from typing import Dict

from arctic_training.config import Config
from arctic_training.register import register_trainer
from arctic_training.trainer import Trainer


class SFTConfig(Config):
    pass


def to_device(batch: Dict, device: str) -> Dict:
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except Exception:
            output[k] = v
    return output


@register_trainer
class SFTTrainer(Trainer):
    config_type = SFTConfig
    dataset_type = "sft"

    def loss(self, batch):
        batch = to_device(self.train_batch_data, self.device)
        outputs = self.model(**batch, use_cache=False)
        loss = outputs.loss
        return loss
