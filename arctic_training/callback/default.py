# Move these to arctic_training.callbacks
def checkpoint_callback(self):
    if self.model.is_gradient_accumulation_boundary():
        if (
            self.model.global_steps >= 1
            and self.model.global_steps % self.config.ckpt_save_interval == 0
        ):
            self.checkpoint_engine.save(with_optimizer=False)


def validate_callback(self):
    if self.model.is_gradient_accumulation_boundary():
        if (
            self.config.eval_frequency > 0
            and (self.model.global_steps + 1) % self.config.eval_frequency == 0
        ):
            self.validate()


def earlystop_callback(self):
    if self.model.is_gradient_accumulation_boundary():
        if self.model.global_steps >= self.get_training_horizon():
            self.early_stop = True
