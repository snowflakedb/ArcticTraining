# Activation checkpointing

Compares four activation-checkpointing strategies on a packed-sequence SFT workload,
measuring how long a sequence each one admits before OOM (and throughput at a fixed
length).

The two strategies being swept are independent boolean axes, so each model gets the
full 2x2 grid:

| arm | `packed_seq_tiled_remat` | `native_activation_checkpoint_cpu_offload` | what it does |
|---|---|---|---|
| `gc` | false | false | HF `gradient_checkpointing_enable()` only (baseline) |
| `offload` | false | true | DeepSpeed's native non-reentrant CPU-offload checkpoint engine replaces HF's |
| `tiling` | true | false | token-axis tiled rematerialization on each layer's attention mixer, HF checkpointing still on underneath |
| `tiling-offload` | true | true | both |

Tiling shrinks each layer's mixer workspace; the native engine moves saved activations
to CPU. They act on different things, so `tiling-offload` tests whether the savings
compose.

## Models

| model | attention | why it's here |
|---|---|---|
| `Qwen/Qwen3-8B` | dense, full attention only | the straightforward case — `self_attn` on every layer |
| `Qwen/Qwen3.5-27B` | hybrid: GDN `linear_attn` + full attention | stresses the tiling path hardest, and is the model where the tiling-vs-gc memory-wall anomaly showed up |

Eight YAMLs total: `run-h200-sp1-<model>-<arm>.yml`.

## The shared baseline

Every arm, for both models, carries the same four activation optimizations. They are
*always on* — they are the floor the checkpointing comparison is measured on top of,
not variables:

| baseline element | how it's set here | bench-script equivalent |
|---|---|---|
| flash attention | `model.attn_implementation: flash_attention_2` | `--flash` |
| tiled MLP | `tiled_mlp_compute: true` | `--tiled_mlp_shards 4` |
| fused linear cross-entropy | implied by `model.type: "liger"` — see below | `--fused_ce` |
| Liger kernels | `model.type: "liger"` | `--fused_ce` / liger apply |

Plus ZeRO-2 with CPU optimizer offload, `cpu_adam`, SP=1, and bf16. So each cell
differs only in its checkpointing strategy.

**Fused CE has no config key of its own.** `SFTTrainer.loss()` branches on
`config.model.type == "liger"` and, on that path, lets liger compute fused logits+loss
in one step (no materialized `[B, S, V]` logits tensor);
`apply_liger_kernel_to_qwen3*` defaults `fused_linear_cross_entropy=True`. So
`type: "liger"` is simultaneously the "liger" and the "fused-ce" element of the
baseline — don't look for a `fused_ce:` field, and don't set `model.type` to anything
else here, or the trainer silently falls back to its own tiled logits+loss path and the
baseline is no longer the one the bench numbers were taken on.

`tiled_mlp_compute` tiles the *MLP*, a different module from the attention mixer that
`packed_seq_tiled_remat` tiles, so the two are orthogonal and compose.
`LigerModelFactory` passes `swiglu=False` to liger when `tiled_mlp_compute` is on, so
liger's swiglu override and ArcticTraining's MLP tiling don't race over the same class.

## Why ZeRO-2 and not ZeRO-3

ZeRO-2 here is a correctness requirement, not a preference.

Under ZeRO-3 the tiled forward calls each mixer once per tile, so DeepSpeed re-runs its
per-submodule hooks per tile and registers the same parameter into a gradient bucket
twice — tripping the `ds_id` uniqueness assert in `stage3.py`
(`assert len(set(p.ds_id for p in params_in_bucket)) == len(params_in_bucket)`). The
bench script avoids this by calling `set_z3_leaf_modules()` on the tiled mixer classes
(and on `GatedDeltaNet` for Qwen3.5), labelling it "required, not an optimization".
**ArcticTraining never calls `set_z3_leaf_modules`.** ZeRO-2 has no equivalent hazard —
the whole leaf-marking requirement is gated on `zero_stage == 3`.

ZeRO-2 also matches the baseline the bench sweeps were actually run on, which makes
these numbers comparable to the ones already collected.

Note the tradeoff: ZeRO-2 cannot offload parameters (that is stage-3 only), so params
stay resident on every GPU and there is materially less activation headroom than in the
stage-3 configs under `../sequence-parallelism/`. The `max_length` starting guesses here
are set well below those accordingly.

One caveat if anyone revisits this: a single-GPU run will *not* reproduce the ZeRO-3
failure, because params are not partitioned across ranks at `--num_gpus 1` and the
grad-bucket path never executes. Validating a stage-3 variant requires a real multi-GPU
run.

## Requires the DeepSpeed fork

Both new axes depend on modules that do **not** exist in released DeepSpeed:

- `deepspeed.runtime.activation_checkpointing.tiled_rematerialization`
- `deepspeed.runtime.activation_checkpointing.offload_activations`

They come from the `sfc-gh-truwase/DeepSpeed` fork, branch
`docs/cpu-checkpointing-perf-note` or newer. Put that checkout ahead of the installed
`deepspeed` on `PYTHONPATH` **before** invoking `arctic_training` — the DeepSpeed
launcher passes the parent environment through to each rank, so exporting it in the
launching shell is enough for a single-node job:

```bash
export PYTHONPATH="/path/to/DeepSpeed-fork:$PYTHONPATH"
arctic_training run-h200-sp1-qwen3.5-27b-tiling-offload.yml --num_gpus 8
```

Run the `gc` arm under the same shadow as the other three, so `deepspeed.__file__` is
identical across all four and the comparison isolates the flags rather than the
DeepSpeed version.

If the fork isn't on the path, both wiring entry points raise `ImportError` naming the
export above rather than failing obscurely later.

## Other environment requirements

- **`liger-kernel>=0.8.2`** for Qwen3.5 — earlier versions have no
  `apply_liger_kernel_to_qwen3_5`, and `AutoLigerKernelForCausalLM` will raise that the
  architecture is unsupported.
- **A transformers build with Qwen3.5 support** — `Qwen3_5Config` is not in released
  transformers; the bench sweeps pin a git revision for it.
- **`causal-conv1d` and `fla` / `flash-linear-attention`** for the Qwen3.5 *tiling*
  arms specifically — see below. Not needed for Qwen3-8B or for the `gc`/`offload`
  arms.

## GDN state and the Qwen3.5 tiling arms

Qwen3.5's `linear_attn` layers carry recurrent and short-conv state that must reset at
every packed-sample boundary. The wiring checks for this and **refuses to run**
(`RuntimeError` from `_gdn_boundary_reset_gaps`) rather than tiling anyway, if the
resolved kernels would drop the `cu_seqlens` / `seq_idx` kwargs that perform those
resets — which is what happens when `causal-conv1d` or `fla` is missing and
transformers falls back to a torch implementation whose signature omits them.

Refusing is deliberate: the untiled packed baseline already leaks state across samples
in that situation, and tiling would reset only at *tile* boundaries, producing numbers
that match neither the baseline nor the truth. If a tiling arm raises this, install the
kernels — don't work around it.

## Packing is required

`pack_samples: true` (a top-level `data:` field, not a per-source one) and
`micro_batch_size: 1` — ArcticTraining asserts the latter when packing is on.

Tiling specifically needs packed input: it tiles along the token axis at packed-sample
boundaries, which it derives from `cu_seqlens`. ArcticTraining's SFT collator emits only
`position_ids` for packed rows, so the wiring derives `cu_seqlens` from those via
transformers' own `prepare_fa_kwargs_from_position_ids`. The first time that fallback
fires it logs:

```
[arctic_training] packed_seq_tiled_remat: derived cu_seqlens from position_ids ...
```

## Results so far

**Qwen3-8B, 8xH200, ZeRO-2** — largest `max_length` that fits, binary-searched
2026-10-15. Each arm's value is recorded inline in its YAML.

| arm | max_length | vs `gc` | OOM at |
|---|---|---|---|
| `gc` | 311,296 | — | 327,680 |
| `offload` | 118,784 | **-62%** | 122,880 |
| `tiling` | **360,448** | **+15.8%** | 376,832 |
| `tiling-offload` | 196,608 | -36.8% | 204,800 |

Every trial logged its `engaged:` markers, so all four arms genuinely exercised their
code paths — including the `position_ids` -> `cu_seqlens` fallback on the tiling arms.

Two findings:

**Tiling works.** +15.8% over the baseline. The gain comes from capping the attention
mixer's workspace at the token budget; it does *not* touch the per-layer checkpoint
boundary tensors (36 x 4096 x 2 B = 288 KiB/token for this model), which remain the
dominant term and set the overall scale of the wall.

**Offload is broken** — see the next section.

Qwen3.5-27B is not measured yet; it is deliberately blocked on the offload fix, since at
~54 GB of weights per trial it would mostly pay to reproduce a known bug.

## The offload regression (unresolved)

`native_activation_checkpoint_cpu_offload` currently makes memory **worse**, not better:
118,784 vs the gc baseline's 311,296, a 62% loss. This inverts the feature's purpose, so
treat both offload-bearing arms' numbers as measurements of a defect, not as a property
of CPU offloading.

What is established:

- The engine really is active — runs log
  `native_activation_checkpoint_cpu_offload enabled on N layers`. Not an inert flag.
- At the OOM point ~137 GiB of 139.8 GiB is already allocated, and the failing
  allocation is inside liger's `fused_linear_cross_entropy_forward` — i.e. the model got
  all the way through the decoder stack still holding nearly everything, and died at the
  loss. The checkpointed activations are not leaving the GPU.
- Module aliasing is *not* the cause: `deepspeed/__init__.py` does
  `from .runtime.activation_checkpointing import checkpointing`, so the
  `deepspeed.checkpointing` that `configure(checkpoint_in_cpu=True)` is called on is the
  same module object whose `_cpu_offload_engine` the wiring sets.
- Tiling composes with it normally (`tiling-offload` is +65% over `offload` alone), which
  localizes the fault to the offload path rather than an interaction.

Remaining suspects: `CPU_CHECKPOINT` not actually true at layer-execution time (it gates
the engine lookup at `checkpointing.py:868`), or the engine's stream / pinned-buffer path
not releasing the GPU-side copies.

## Tuning `max_length`

Every `max_length` here is a starting guess, not a measured value. These arms exist to
find the memory wall, so binary-search each one for the largest value that fits and
record it inline, matching the convention in `../sequence-parallelism/` (e.g.
`max_length: 650_000 # ok 126.81 GB`).

## Verifying an arm engaged

Each arm prints a one-line confirmation at startup:

```
[arctic_training] packed_seq_tiled_remat token_budget=8192 classes=['Qwen3Attention'] modules=36
[arctic_training] native_activation_checkpoint_cpu_offload enabled on 36 layers
```

A missing line means the flag didn't take effect — treat the run as invalid rather than
reporting its number. Both entry points also raise if they match zero modules.
