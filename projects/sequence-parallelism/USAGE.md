# Ulysses Sequence Parallelism for HF Transformers Usage

## Configuration

Define the desired sequence parallelism degree in the config yaml file with:
```
sequence_parallel_size: 8
```
The degree of sequence parallelism shouldn't exceed the total number of gpus.

You can use fewer than the total number of gpus for sequence parallelism, as long as the number is of base 2, for example with 8 gpus you could do:
```
sequence_parallel_size: 4
```
which would lead to 2 replicas.


## component versions

Here are some known component versions that are likely to impact the max achievable seqlen.

### pytorch

- there is 4GB leak in torch.distributed when `barrier` is called on gpu0 w/ 8 gpu-nodes - it doesn't exist in pt-2.4, but is there in 2.6*, 2.7.0. The bug fix will appear in 2.7.1 and 2.8.
- if you add `all_gather_object` with tensors placed on cuda, that would waste ~4GB/per gpu on any pytorch version up to 2.7.x - perhaps it'll get fixed in the future to do the right thing.


## Requirements

### Batch seqlen

The batch seqlen has to be divisible by sequence parallel size.



## Performance

### ChunkedMemEfficientLoss

Normally when loss is calculated for a long seqlen it consumed a huge amount of memory. For example, let's take seqlen=131072 (128K) - when split 8-way, each rank will compute a seqlen shard of 16384. Now with [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json) that has a vocab of `128_256` at fp32 this is:

```
18750*4*128256/2**30 = 7.82GB
```

Let's round it up to 8GB. Let's next look at the memory profiler for the loss function:

![loss calculation normal](images/)

![loss calculation sharded](images/)


### Batch seqlen

While the batch seqlen has to be divisible by sequence parallel size, ideally you also want it to be divisible by a largish `2**x` - e.g. 256 to have the fastest matrix multiplication.

So together, try to have it divisible by `256*sequence_parallel_size` if it works out, if not any `2**x*sequence_parallel_size` (where `x>=0`) will do functionally-wise.

## Performance vs. Correctness

The longer the sequence the higher the performance will be.

### Longer Sequence Length at the potential loss of correctness

By default deepspeed will use fp32 for sequence parallelism communications.

If you set `"reduce_bucket_size": 5e8` in deepspeed config (default) - at bf16 (2 bytes per param) this will use 1GB of memory to reduce gradients (`2*0.5**9`). But at fp32 (4 bytes per param) this will require additional 4GB of memory:
1. 2GB to copy bf16 to fp32 grads
2. 2GB to reduce them

So if you need to push the sequence length and want to shave some GBs off to accomodate a longer sequence length you can do 2 things:

1. sacrify some of the correctness by forcing the sequence parallelism reductions to be performed in bf16, by setting:
```
"seq_parallel_communication_data_type": 'bf16'
```
in the deepspeed config.

2. or you can sacrify some speed, by reducing the `reduce_bucket_size` - for example, if you set it to:
```
"reduce_bucket_size": 1.25e8
```
which makes the reduction bucket 4x smaller than `5e8` you will incur only 1GB of additional GPU memory to perform the reduction instead of 4GB, because now you have
1. 0.5GB to copy bf16 to fp32 grads
2. 0.5GB to reduce them
