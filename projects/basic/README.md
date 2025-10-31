# Basic functionality recipes

## Causal Trainer

This is a basic non-instruct trainer, with a plain text dataset `CausalDataset` (e.g. books) driven by `CausalDataFactory` that slices the text into `max_length` chunks and if there are any chunks shorter than `max_length` these left-overs go into packed sequences. So that some samples are made from a slice of a single sample, others are 2 or more short slices. The `CausalDataFactory` dataset packer generates them as they come, ending up with a series of `max_length` samples, followed by 1 packed sample, then the cycle repeats.

Here is how you define the datasource for this dataset:
```
data:
  sources:
    - type: CausalDataset
      name_or_path: manu/project_gutenberg
      split: en
      sample_count: 100_000
```
or if you want to use the magic syntax sugar:
```
data:
  sources:
    - type: CausalDataset
      name_or_path: manu/project_gutenberg:en[:100]
```

Here is a 1-gpu recipe to try:
```
arctic_training run-causal.yml
```
