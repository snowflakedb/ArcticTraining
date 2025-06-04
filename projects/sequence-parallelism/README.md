# Ulysses Plus - very long sequence length post-training

Ulysses Plus enables very long sequence length post-training using out-of-the-box HuggingFace Transformers models. There are two parts to Ulysses Plus:

1. Ulysses Sequence Parallelism for HF Transformers implements an efficient way of processing long sequences by employing sequence parallelism and attention head parallelism.
2. Ulysses Plus enables even longer sequence lengths using multiple tricks such as activation checkpoint offload to CPU, tiled MLP and logit+loss compute and `PYTORCH_CUDA_ALLOC_CONF` optimizations.

Where to do next:

1. If you want to jump right into trying it out proceed to [SFT Training examples](#sft-training-examples).
2. To go into more details refer to [USAGE.md](USAGE.md).
3. To integrate UlyssesPlus into a different framework see: https://www.deepspeed.ai/tutorials/ulysses-plus-sequence-pallellism

## SFT Training examples

### Preliminaries

Since LLama-3 models are gated to try Llama 8B/70B examples you need to ensure you can access those models first. In the following instructions we assume you already have `HF_TOKEN` environment variable setup so that you could access the gated LLama models. If you don't have access, go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct and request access. If you don't have a token follow [these instructions](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication). If you find it too complicated just skip the LLama examples and use Qwen3 examples which aren't gated.

### Install ArcticTraining and its prerequisites

```
git clone https://github.com/snowflakedb/ArcticTraining
cd ArcticTraining
pip install .
cd projects/sequence-parallelism
```

### 1 GPU

To launch a 1-GPU job:
```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
arctic_training run-sp1-llama-8b.yml
```

You have 2 examples for 1 gpu:

- [run-sp1-llama-8b.yml](run-sp1-llama-8b.yml)
- [run-sp1-qwen3-32b.yml](run-sp1-qwen3-32b.yml)

### 1 node

To launch an 8-GPU job:
```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
arctic_training run-sp8-llama-8b.yml
```

You have multiple examples for 1 node:

- [run-sp8-llama-8b-no-mlp-no-act-offload.yml](run-sp8-llama-8b-no-mlp-no-act-offload.yml)
- [run-sp8-llama-8b-no-ulysses-no-liger-no-extras.yml](run-sp8-llama-8b-no-ulysses-no-liger-no-extras.yml)
- [run-sp8-llama-8b-no-ulysses-yes-liger-no-extras.yml](run-sp8-llama-8b-no-ulysses-yes-liger-no-extras.yml)
- [run-sp8-llama-8b-yes-act-offload.yml](run-sp8-llama-8b-yes-act-offload.yml)
- [run-sp8-llama-8b-yes-mlp.yml](run-sp8-llama-8b-yes-mlp.yml)
- [run-sp8-llama-8b.yml](run-sp8-llama-8b.yml)
- [run-sp8-llama-70b.yml](run-sp8-llama-70b.yml)
- [run-sp8-qwen3-32b.yml](run-sp8-qwen3-32b.yml)

### 2+ nodes

Running a multi-node job is slightly more complicated than a single node job as you need to setup the same environment for all participating nodes and get them to run in sync.

1. Find your `hostfile`

Here we assume you have a `/etc/hostfile` with contents like:

```
nodename1: slots=8
nodename2: slots=8
```
this was a 2-node example. This file could be existing at a different location. You can also create one yourself. For nuances see [Resource Configuration (multi-node)](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

If you are using SLURM, `srun` already knows which nodes it'll run on, so there you don't need the `hostfile` - you can check some of the launcher examples [here](https://github.com/stas00/ml-engineering/blob/master/orchestration/slurm/launchers/README.md).

2. Prepare a shared between hosts environment variables file:

```
echo HF_HOME=/checkpoint/huggingface                              > .deepspeed_env
echo HF_TOKEN=$HF_TOKEN                                          >> .deepspeed_env
echo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True            >> .deepspeed_env
```

`.deepspeed_env` is a special filename, for details and how to override its default name please see [this](https://www.deepspeed.ai/getting-started/#multi-node-environment-variables)

This file is also a good place to add any environment variables required by your CSP to get the intra-node networking work propertly if it wasn't preset at node launch time.

3. Run the experiment:

To run the 4-node job:
```
arctic_training run-sp32-llama-70b.yml -H /etc/hostfile
```
after editing the path to the `hostfile` you created or found in step 1.

You have multiple examples for 2+ node jobs:

2 nodes:
- [run-sp16-llama-8b.yml](run-sp16-llama-8b.yml)
- [run-sp16-llama-70b.yml](run-sp16-llama-70b.yml)
- [run-sp16-qwen3-32b.yml](run-sp16-qwen3-32b.yml)

4 nodes:
- [run-sp32-llama-8b.yml](run-sp32-llama-8b.yml)
- [run-sp32-llama-70b.yml](run-sp32-llama-70b.yml)
- [run-sp32-qwen3-32b.yml](run-sp32-qwen3-32b.yml)

8 nodes:
- [run-sp64-llama-70b.yml](run-sp64-llama-70b.yml)
- [run-sp64-qwen3-32b.yml](run-sp64-qwen3-32b.yml)

### Modifying the dataset

If you want to use your own database, edit this section of the desired `yaml` file and replace it with the dataset of your choice.

```
data:
  sources:
    - HuggingFaceH4/ultrachat_200k
```

### Environment used to create these examples

Hardware:
- H100 80GB nodes w/ 8 GPUs per node
- AWS EFA v2
- 1.9TB CPU memory

Software:
- `torch==2.8.0.dev20250507` (aka nightly) (we found that earlier versions either had memory leaks or weren’t as memory-efficient - 2.7.1 and 2.8 should be good as well when they come out)
- `flash_attn==2.7.4` (but 2.6.4+ is the same performance)
- `transformers=4.51.3`
- `deepspeed=` XXX? Update once PR merged and new version released


### GPU and CPU OOM

If your setup is different from the one that was used to create these examples you may not have the same hardware and software setup. Therefore, if you get a GPU or a CPU OOM lower `max_length` to a smaller value.

Usually it's good to start with something like `max_length: 8_000` to ensure everything else works and then make it much larger.
