# Style Guide

XXX: this is a work in progress - will re-org/group better as we have more items - for now just add things flat

## dict

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
