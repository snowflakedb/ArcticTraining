# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import torch
from deepspeed.utils.timer import SynchronizedWallClockTimer

from arctic_training.trainer.flops_counter import estimate_decoder_transformer_tflos
from arctic_training.utils import human_format_base10_number
from arctic_training.utils import human_format_secs

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer


def _gather_object(value: Union[float, int, list], world_size: int) -> List[float]:
    """All-gather a scalar or list across ranks, returning a flat list."""
    output: list = [None] * world_size
    torch.distributed.all_gather_object(output, value)
    return [v for item in output for v in (item if isinstance(item, list) else [item])]


def _compute_tflos(metrics: Metrics, ctx: Dict) -> Optional[float]:
    """Compute and cache tflos_total from raw seqlens in the accumulator."""
    if "tflos_total" in ctx:
        return ctx["tflos_total"]
    seqlens_raw = metrics._accum.get("seqlens", [])
    if not seqlens_raw:
        return None
    batch_of_seqlens = [s for batch in seqlens_raw for s in batch]
    ws = metrics.trainer.world_size
    tflos_sub, _ = estimate_decoder_transformer_tflos(
        hf_model_config=metrics.trainer.model_unwrapped.config,
        model_size=metrics._model_size,
        batch_of_seqlens=batch_of_seqlens,
        enable_gradient_checkpointing=not metrics.trainer.config.model.disable_activation_checkpoint,
    )
    ctx["tflos_total"] = sum(_gather_object(tflos_sub, ws)) / metrics.trainer.config.sequence_parallel_size
    return ctx["tflos_total"]


def _derive_tflops(time_key: str) -> Callable:
    """Factory for tflops derive functions: tflos_total / raw gathered time total."""

    def _derive(metrics: Metrics, ctx: Dict) -> Optional[float]:
        tflos = _compute_tflos(metrics, ctx)
        time_total = ctx.get(f"{time_key}_total")
        if tflos and time_total:
            return tflos / time_total
        return None

    return _derive


def _compute_interval_token_sum(metrics: Metrics) -> Optional[int]:
    """Total tokens since last log (summed over all steps and DP)."""
    seqlens_raw = metrics._accum.get("seqlens", [])
    if not seqlens_raw:
        return None
    local_total = sum(s for batch in seqlens_raw for sublist in batch for s in sublist)
    gathered = _gather_object(local_total, metrics.trainer.world_size)
    return sum(gathered)


def _derive_seqlen(metrics: Metrics, ctx: Dict) -> Optional[float]:
    """Average seqlen per step per DP (since last log)."""
    total_tokens = _compute_interval_token_sum(metrics)
    if total_tokens is None:
        return None
    seqlens_raw = metrics._accum.get("seqlens", [])
    ws = metrics.trainer.world_size
    num_steps = len(seqlens_raw) * ws
    return total_tokens / num_steps if num_steps else None


def _derive_seqlen_total_since_log(metrics: Metrics, ctx: Dict) -> Optional[int]:
    """Total tokens since last log (all steps, all DP ranks)."""
    return _compute_interval_token_sum(metrics)


class MetricDef:
    """Defines how a metric is reduced, formatted, and displayed."""

    __slots__ = ("reduce", "fmt", "display_name", "wandb", "accumulate", "derive")

    def __init__(
        self,
        reduce: str = "mean",
        fmt: Optional[Union[str, Callable]] = None,
        display_name: Optional[str] = None,
        wandb: bool = True,
        accumulate: bool = False,
        derive: Optional[Callable] = None,
    ):
        self.reduce = reduce
        self.fmt = fmt
        self.display_name = display_name
        self.wandb = wandb
        self.accumulate = accumulate
        self.derive = derive

    def format_value(self, value: Union[int, float]) -> str:
        if callable(self.fmt):
            return self.fmt(value)
        if isinstance(self.fmt, str):
            return f"{value:{self.fmt}}"
        return str(value)


class Metrics:
    """Tracks, accumulates, and reports training metrics with GAS support.

    All values accumulate continuously. When ``report()`` is called, the
    ``accumulate`` flag on each metric controls which values are used:

    - ``accumulate=False`` (default): Only the latest value is used.
    - ``accumulate=True``: All values since the previous ``report()`` are
      aggregated, giving averages over the full log interval.

    After reporting, all accumulators are cleared.

    Metrics can also be *derived* — computed from other metrics via a
    ``derive(metrics, ctx)`` callable rather than recorded directly.

    Standard metrics are registered by default. Trainers can register
    additional metrics via ``register()``.
    """

    def __init__(self, trainer: Trainer) -> None:
        self.enabled = trainer.config.train_log_iter_interval > 0
        if not self.enabled:
            return

        self.trainer = trainer
        self._defs: Dict[str, MetricDef] = {}
        self._accum: Dict[str, list] = defaultdict(list)
        self._timers: Dict[str, SynchronizedWallClockTimer.Timer] = {}
        self._display_order: List[str] = trainer.config.metrics_display_order
        self.summary_dict: Dict[str, Union[int, float]] = {}

        # Register standard metrics -- display order follows registration order.
        self.register("epoch", derive=lambda m, _: m.trainer.epoch_idx, display_name="epoch")
        self.register("loss", reduce="mean", fmt=".4f", display_name="loss")
        self.register("eval_loss", reduce="mean", fmt=".4f", display_name="eval loss")
        self.register("lr", derive=lambda m, _: m.trainer.model.lr_scheduler.get_last_lr()[0], fmt=".3E", display_name="lr")
        self.register("seqlens", derive=_derive_seqlen, fmt=human_format_base10_number, display_name="seqlen (avg/step/DP)")
        self.register("seqlen_total_since_log", derive=_derive_seqlen_total_since_log, fmt=human_format_base10_number, display_name="seqlen total (since log)")
        self.register("step_time", reduce="mean", fmt=human_format_secs, display_name="step time", accumulate=True)
        self.register("step_tflops", derive=_derive_tflops("step_time"), fmt=".1f", display_name="step tflops")
        self.register("iter_time", reduce="mean", fmt=human_format_secs, display_name="iter time", accumulate=True)
        self.register("iter_tflops", derive=_derive_tflops("iter_time"), fmt=".1f", display_name="iter tflops")
        self.register("mem_ma", reduce="mean", fmt=lambda v: f"{v:.2f} GB", display_name="MA")
        self.register("mem_max_ma", reduce="mean", fmt=lambda v: f"{v:.2f} GB", display_name="Max_MA")
        self.register("mem_nv", reduce="mean", fmt=lambda v: f"{v:.2f} GB", display_name="NV")

        numel = lambda p: p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
        self._model_size = sum(numel(p) for p in trainer.model_unwrapped.parameters())

        horizon = trainer.training_horizon
        if trainer.config.exit_iteration > 0:
            horizon = min(trainer.config.exit_iteration, horizon)
        self._max_iter = horizon
        self._max_iter_pad = len(str(horizon))

    def register(
        self,
        name: str,
        reduce: str = "mean",
        fmt: Optional[Union[str, Callable]] = None,
        display_name: Optional[str] = None,
        wandb: bool = True,
        accumulate: bool = False,
        derive: Optional[Callable] = None,
    ) -> None:
        """Register a new metric (or override an existing one).

        Args:
            name: Key used with ``record()``.
            reduce: ``"mean"`` or ``"sum"`` — how to reduce across GAS micro-steps.
            fmt: Format spec string (e.g. ``".4f"``) or callable for display.
            display_name: Label shown in the log line. ``None`` hides it from display.
            wandb: Whether to include in wandb logs.
            accumulate: If ``True``, ``report()`` aggregates all values since
                the previous report. If ``False`` (default), only the latest
                GAS cycle's values are used.
            derive: A callable ``(metrics, ctx) -> value`` for metrics computed
                from other metrics rather than recorded directly. *metrics* is the
                ``Metrics`` instance (gives access to ``_accum``, ``trainer``,
                etc.). *ctx* contains ``{key}_total`` (raw gathered sum) for
                every reduced metric.
        """
        if not self.enabled:
            return
        self._defs[name] = MetricDef(
            reduce=reduce, fmt=fmt, display_name=display_name, wandb=wandb, accumulate=accumulate, derive=derive
        )

    def record(self, key: str, value) -> None:
        """Record a metric value. Always appends to the accumulator."""
        if not self.enabled:
            return
        self._accum[key].append(value)

    def start_timer(self, key: str) -> None:
        """Start (or create) a wall-clock timer."""
        if not self.enabled:
            return
        if key not in self._timers:
            self._timers[key] = SynchronizedWallClockTimer().Timer(key)
        self._timers[key].start()

    def stop_timer(self, key: str) -> None:
        """Stop a timer and accumulate its elapsed time (seconds)."""
        if not self.enabled:
            return
        if key not in self._timers:
            raise KeyError(f"Timer {key} not started")
        self._timers[key].stop()
        self._accum[f"{key}_time"].append(self._timers[key].elapsed() / 1000)

    def restart_timer(self, key: str) -> None:
        """Stop and immediately restart a timer."""
        self.stop_timer(key)
        self.start_timer(key)

    def should_log(self) -> bool:
        """Whether metrics should be logged on the current global step."""
        return (
            self.enabled
            and self.trainer.global_step > 0
            and self.trainer.global_step % self.trainer.config.train_log_iter_interval == 0
        )

    def clear(self) -> None:
        """Clear all accumulated values."""
        if not self.enabled:
            return
        self._accum.clear()

    def report(self, prefix: str = "train") -> Dict[str, Union[int, float]]:
        """Reduce, gather, print, and return the metrics summary.

        For ``accumulate=False`` metrics, only the latest GAS cycle's values
        are used. For ``accumulate=True`` metrics, all values since the last
        report are aggregated.

        After reporting, all accumulators are cleared.
        """
        if not self.enabled:
            return {}

        self.summary_dict.clear()
        self.summary_dict["iter"] = self.trainer.global_step
        ws = self.trainer.world_size

        # Context dict passed to derive functions -- populated below with
        # raw gathered totals ({key}_total) for every reduced scalar metric.
        ctx: Dict[str, float] = {}

        # Reduce each scalar accumulator and gather across ranks.
        for key, values in list(self._accum.items()):
            if not values or not isinstance(values[0], (int, float)):
                continue
            defn = self._defs.get(key, MetricDef())
            if not defn.accumulate:
                values = [values[-1]]
            local_sum = sum(values)
            gathered = _gather_object(local_sum, ws)
            total = sum(gathered)
            if defn.reduce == "mean":
                self.summary_dict[key] = total / (len(values) * len(gathered))
            else:
                self.summary_dict[key] = total / len(gathered)
            ctx[f"{key}_total"] = total

        # Evaluate derived metrics
        for key, defn in self._defs.items():
            if defn.derive is not None and key not in self.summary_dict:
                value = defn.derive(self, ctx)
                if value is not None:
                    self.summary_dict[key] = value

        if self.trainer.global_rank == 0:
            self._print(prefix)

        self.clear()
        return dict(self.summary_dict)

    def _print(self, prefix: str) -> None:
        s = self.summary_dict
        parts = [
            f"{prefix.title():>{len('train')}} iter:"
            f" {s['iter']:>{self._max_iter_pad}}/{self._max_iter}"
            f" {100 * s['iter'] // self._max_iter:>3}%"
        ]
        if self._display_order:
            keys = list(self._display_order) + [k for k in self._defs if k not in self._display_order]
        else:
            keys = list(self._defs)
        for key in keys:
            defn = self._defs.get(key)
            if defn and defn.display_name and key in s:
                parts.append(f"{defn.display_name}: {defn.format_value(s[key])}")
        print(" | ".join(parts))
