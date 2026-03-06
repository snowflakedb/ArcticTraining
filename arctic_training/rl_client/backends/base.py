"""Backend -- unified transport protocol for inference and training servers.

ArcticRLClient delegates all server communication to a Backend.
Swap the backend to switch between direct ArcticInference server access
(short-term) and dss-platform access (long-term) without changing any
RL-framework code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator

import torch


class Backend(ABC):
    """Transport-agnostic contract for both inference and training operations.

    Every method returns plain dicts/lists so the caller is decoupled
    from HTTP details, job-id schemes, or serialisation formats.
    """

    # ------------------------------------------------------------------
    # Inference: Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def init_model(
        self,
        model_config: dict[str, Any],
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Load a model and return at least ``{"model_id": ...}``."""

    @abstractmethod
    def shutdown_model(self, model_id: str) -> dict[str, Any]:
        """Unload a single model."""

    @abstractmethod
    def shutdown(self) -> dict[str, Any]:
        """Shut down the entire server."""

    # ------------------------------------------------------------------
    # Inference: Generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        model_id: str,
        prompts: list[str] | None = None,
        *,
        prompt_token_ids: list[list[int]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate completions.  Returns a list of per-prompt result dicts."""

    # ------------------------------------------------------------------
    # Inference: Weight sync
    # ------------------------------------------------------------------

    @abstractmethod
    def update_weights(
        self,
        model_id: str,
        groups: list[dict[str, Any]],
        *,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        direct_mode: bool = False,
    ) -> dict[str, Any]:
        """Trigger NCCL weight receive on the inference server."""

    @abstractmethod
    def close_weight_sync(self, model_id: str) -> dict[str, Any]:
        """Destroy persistent NCCL receiver engines."""

    # ------------------------------------------------------------------
    # Inference: Wake / Sleep
    # ------------------------------------------------------------------

    @abstractmethod
    def wake_up(self, model_id: str) -> dict[str, Any]:
        """Resume workers from sleep."""

    @abstractmethod
    def sleep(self, model_id: str, level: int = 1) -> dict[str, Any]:
        """Release GPU memory (weights + KV cache)."""

    # ------------------------------------------------------------------
    # Inference: Status
    # ------------------------------------------------------------------

    @abstractmethod
    def status(self) -> dict[str, Any]:
        """Server / model status."""

    @abstractmethod
    def weights_info(self, model_id: str) -> dict[str, Any]:
        """Parameter metadata for a loaded model."""

    # ------------------------------------------------------------------
    # Training: Forward / backward
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_log_prob(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Forward-only pass returning per-token log probabilities."""

    @abstractmethod
    def update_policy(
        self,
        batch: dict[str, Any],
        loss_type: str = "ppo",
        loss_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward + backward + optimizer step.  Returns training metrics."""

    # ------------------------------------------------------------------
    # Training: Weight access
    # ------------------------------------------------------------------

    @abstractmethod
    def get_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over un-sharded model parameters."""

    # ------------------------------------------------------------------
    # Training: Checkpointing
    # ------------------------------------------------------------------

    @abstractmethod
    def save_checkpoint(self, path: str) -> dict[str, Any]:
        """Persist model + optimizer state to *path*."""

    @abstractmethod
    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """Restore model + optimizer state from *path*."""
