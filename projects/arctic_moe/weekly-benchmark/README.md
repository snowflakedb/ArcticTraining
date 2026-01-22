# Weekly benchmarks

## Qwen3Moe

```
./run-dp8-ep1-qwen3-30b-dense.sh
./run-dp8-ep8-qwen3-30b.sh
```

Config diff sanity check:
```
diff -u run-dp8-ep8-qwen3-30b.yml run-dp8-ep1-qwen3-30b-dense.yml
```


## Qwen3Next


```
./run-dp8-ep1-qwen3-next-80b-dense.sh
./run-dp8-ep8-qwen3-next-80b.sh
```

Config diff sanity check:
```
diff -u run-dp8-ep8-qwen3-next-80b.yml run-dp8-ep1-qwen3-next-80b-dense.yml
```

- Sequence equivalent:
`Qwen3-Next-80B` on 8 gpus

```
num_experts=512
num_experts_per_tok=10
dense_seqlen=16_000
n_gpus=8
```
thus deriving:
```
equivalent_moe_seqlen = 102_400
```

So we need to 6.4x seqlen for the AMoE to match the dense equivalent seqlen (edited)



## Calculating the seqlen equivalent

The goal is to get the same compute intensity for the dense and moe versions - so that each token goes through the same number of

```
effective_seqlen_per_gemm = seqlen * n_gpus / (num_experts / num_experts_per_tok)
```

thus inverting to derive our need:

```
equivalent_moe_seqlen = dense_seqlen * (num_experts / num_experts_per_tok) / n_gpus
```

- `Qwen3-30B` on 8 gpus:
```
num_experts=128
num_experts_per_tok=8
dense_seqlen=16_000
n_gpus=8
```
thus deriving:
```
equivalent_moe_seqlen = 32_000
```

So we need to 2x seqlen for the AMoE to match the dense equivalent seqlen


- `Qwen3-Next-80B` on 8 gpus

```
num_experts=512
num_experts_per_tok=10
dense_seqlen=16_000
n_gpus=8
```
thus deriving:
```
equivalent_moe_seqlen = 102_400
```

So we need to 6.4x seqlen for the AMoE to match the dense equivalent seqlen (edited)
