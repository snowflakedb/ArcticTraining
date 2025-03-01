# Style Guide

XXX: this is a work in progress - will re-org/group better as we have more items - for now just add things flat

## Code style

### dict

Use `kwargs` style dicts. This:
- leads to need to type less quotes
- allows to quickly copy to/from kwargs in the function and its object assignments right after the method/function is defined

Example: Instead of:
```
cache_path_args = {
    "data_factory_args": {
        "a": 1,
        "b": 2,
    },
    "data_source_args": self.config.model_dump(),
    "tokenizer_factory_args": self.trainer.config.tokenizer.model_dump(),
}
```
use:
```
cache_path_args = dict(
    data_factory_args=dict(a=1, b=2)
    data_source_args=self.config.model_dump(),
    tokenizer_factory_args=self.trainer.config.tokenizer.model_dump(),
)
```

### Unambiguous conditionals

Consider:
```
x = []
if not x: do something
```
This is ambiguous since `x` could be `None` or `[]` or even a boolean, which could lead to difficult to find errors.

A better style is:
```
x = []
if len(x) == 0: do something
```

In other words always check explicitly for the exact condition you're wanting to meet.

This approach also makes the code more readable by someone who didn't write it.


### Warnings

1. We don't want to contribute to the myriads of warnings already emitted by the sub-systems rendering the log stream useless to end user because they can't tell what's important from non-important because there are too many warnings.

   If there is an issue it should assert, if there is an indication it should be info-level log for those who want it.

2. don't issue warnings when a default value is used, this is the whole point of defaults - we just need to make sure they are well documented. The default value could be a function arg default, but it can also be a bigger scope default - e.g. which sub-system was deployed since the user didn't specify any preference. `log.info` is probably OK for that purpose.


## Testing

In general it's recommended to read:

1. [Methodology](https://github.com/stas00/the-art-of-debugging/tree/master/methodology)
2. [PyTest testing](https://github.com/stas00/ml-engineering/tree/master/testing)

### Tiny models

Most tests perform only functional testing - i.e. they just check that some system works correctly. In all those cases the tiniest possible model should suffice - by tiniest we mean a model of 5MB - not 500MB - a few layers, a short dict and a shall hidden_size should be more than enough. See [Quick Iterations, Small Payload](https://github.com/stas00/the-art-of-debugging/tree/master/methodology#quick-iterations-small-payload)

Here is how to [create tiny models](https://github.com/stas00/ml-engineering/blob/c5306e5aa52e9729616c954a1a5aa02784bf612c/debug/make-tiny-models-tokenizers-datasets.md#making-a-tiny-model).

Then upload it to the HF hub (public permissions) and use it in the tests.


### Tiny datasets

Identical to models we need for datasets to take a few seconds to download and load for tests to be fast, so here as well, prefer to make tiny datasets and use those.

Here is how to [create tiny datasets](https://github.com/stas00/ml-engineering/blob/c5306e5aa52e9729616c954a1a5aa02784bf612c/debug/make-tiny-models-tokenizers-datasets.md#making-a-tiny-dataset)

Then upload it to the HF hub (public permissions) and use it in the tests.
