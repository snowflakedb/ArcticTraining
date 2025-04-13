# Ulysses Sequence Parallelism for HF Transformers integration

XXX: the first pass is from the perspective of AT, there will be another pass to do the same for any framework

## config

Define the desired sequence parallelism degree in the config yaml file with:
```
sequence_parallel_size: 8
```

## DataLoader

Currently Ulysses Sequence Parallelism requires samples that use `position_ids` and not `attention_mask`. If your implementation already does that, there is nothing to change.

Down the road we might figure out how to make it work with `attention_mask` but at the moment it doesn't work how HF Transformers has it implemented (because we are sharding batches).

## trainer super class

In theory nothing needs to be changed here, but to explain how we plug sp here:

### model setup:

```
        mpu = UlyssesAttentionHF.register_with_transformers(
            model_name_or_path=self.config.model.name_or_path,
            core_attn_implementation=self.config.model.attn_implementation,
            sequence_parallel_size=self.config.sequence_parallel_size,
            max_length=self.config.data.max_length,
            micro_batch_size=self.config.micro_batch_size,
            seq_length_is_variable=True,
        )
        if self.config.sequence_parallel_size > 1:
            # we are overriding the original core attn implementation with `ulysses` and we have already passed the original core attn implementation to `UlyssesAttentionHF`
            self.config.model.attn_implementation = "ulysses"

        dschf = HfDeepSpeedConfig(self.config.deepspeed)  # noqa: F841
        model_factory = self.config.model.factory(self)
        self.model = model_factory()

        UlyssesAttentionHF.validate_model(
            model=self.model,
            sequence_parallel_size=self.config.sequence_parallel_size,
        )

```
That's allmost everything, now just need to pass `mpu` to `deepspeed`.initialize`, like so:

```
        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            args=self.config,
            lr_scheduler=self.scheduler,
            config=self.config.deepspeed,
            mpu=mpu,
        )
```

in `step`:

```
        if self.config.sequence_parallel_size == 1:
            # this is the original code
            loss = self.loss(batch)
            self.model.backward(loss)
            ...

        else:
            # sp will do backward inside sp_fwd_bwd_loss
            # the returned loss is already averaged across ranks
            loss = self.sp_fwd_bwd_loss(batch)
```

So the trainer subclass needs to have `self.sp_fwd_bwd_loss` - which we need to figure out for SwiftKV

## `sp_fwd_bwd_loss`

You will currently find it in `trainer/sft_trainer.py`. If your loss function is the same as sft then you can just reuse this method, if it's not - let's then talk about the differences and probably further split up `sp_fwd_bwd_loss` into chunks or provide some sort of callbacks.

The original SFT loss is just:

```
    def loss(self, batch) -> torch.Tensor:
        batch = to_device(batch, self.device)
        outputs = self.model(**batch, use_cache=False)
        loss = outputs.loss
        return loss
```
the only things that Ulysses does are:
1. gather data batches from all ranks

2. then for each batch:

a. split each into shards
b. each shard runs fwd + loss + bwd on each rank
c. then loss is averaged

3. average loss again across batches

One nuance is that we have to hack deepspeed to override its GAS functionality so that it will do the right thing wrt GAS with the help of `set_gradient_accumulation_boundary` - the overall logic goes:

```
      self.model.set_gradient_accumulation_boundary(False)
      for batch in batches:
          split and assign each shard to its corresponding processing rank
          fwd + loss + bwd on each shard
          average losses across ranks
      average losses across batches
      self.model.set_gradient_accumulation_boundary(True)
```


## Implementation details

### labels need to be pre-shifted

When using batch sharding one can't let the upstream `loss` function to do the labels shifting. Here is why:

When calculating loss in an unsharded batch we end up with (shift left):

```
input_ids: [1 2 3 4 5 6 7    8   ]
labels   : [1 2 3 4 5 6 7    8   ]
shiftedl : [2 3 4 5 6 7 8 -100]
```

When sharded we lose label 5 once shifted:

```
input_ids: [1 2 3    4] [5 6 7    8]
labels   : [1 2 3    4] [5 6 7    8]
shiftedl : [2 3 4 -100] [6 7 8 -100]
```

So a new API was added in HF transformers to support pre-shifted labels, and then we end up with the correct labels passed to the loss function for each shard:

```
input_ids: [1 2 3 4]  [5 6 7 8]
labels   : [1 2 3 4]  [5 6 7 8]
shiftedl : [2 3 4 5]  [6 7 8 -100]
```
