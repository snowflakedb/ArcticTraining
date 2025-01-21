from torch.optim import AdamW

from arctic_training import register
from arctic_training.optimizer import OptimizerFactory
from arctic_training.trainer.sft_trainer import SFTTrainer


class CPUOptimizerFactory(OptimizerFactory):
    name = "cpu"

    def create_optimizer(self, model, optimizer_config):
        return AdamW(model.parameters(), lr=optimizer_config.learning_rate)


@register
class DummyTrainer(SFTTrainer):
    name = "dummy"
    optimizer_factory_type = [CPUOptimizerFactory]
