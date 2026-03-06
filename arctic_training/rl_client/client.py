"""ArcticRLClient -- unified client for Arctic servers.

All operations go through a single :class:`Backend` instance.  The client
caches ``model_id`` after :meth:`init_model` so callers don't need to pass
it on every call.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator

import torch

from arctic_training.rl_client.backends.base import Backend
from arctic_training.rl_client.backends.direct import DirectBackend
from arctic_training.rl_client.config import ArcticRLClientConfig
from arctic_training.rl_client.weight_sync import WeightSyncCoordinator

logger = logging.getLogger(__name__)


def _build_backend(config: ArcticRLClientConfig) -> Backend:
    """Create the backend implied by *config.inference.backend*."""
    base_url = f"http://{config.inference.host}:{config.inference.port}"
    kind = config.inference.backend

    if kind == "direct":
        return DirectBackend(base_url)

    if kind == "dss":
        from arctic_training.rl_client.backends.dss import DSSBackend
        return DSSBackend(base_url)

    raise ValueError(f"Unknown backend: {kind!r}")


class ArcticRLClient:
    """Framework-agnostic client for Arctic servers.

    Parameters
    ----------
    config : ArcticRLClientConfig
        Connection and topology parameters.
    """

    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self.backend = _build_backend(config)
        self.weight_sync = WeightSyncCoordinator(config.weight_sync)
        self.model_id: str | None = config.inference.model_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_model(
        self,
        model_config: dict[str, Any],
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Load a model on the server.  Caches the returned ``model_id``."""
        data = self.backend.init_model(model_config, model_id=model_id)
        self.model_id = data.get("model_id") or model_id or model_config.get("model", "default")
        data["model_id"] = self.model_id
        return data

    def shutdown_model(self, model_id: str | None = None) -> dict[str, Any]:
        """Unload a single model."""
        return self.backend.shutdown_model(model_id or self.model_id)

    def shutdown(self) -> None:
        """Tear down NCCL senders and shut down the server."""
        self.weight_sync.destroy()
        try:
            self.backend.shutdown()
        except Exception:
            logger.warning("Server shutdown failed", exc_info=True)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str] | None = None,
        *,
        prompt_token_ids: list[list[int]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate text completions."""
        return self.backend.generate(
            self.model_id,
            prompts,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def update_weights(
        self,
        groups: list[dict[str, Any]],
        *,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        direct_mode: bool = False,
    ) -> dict[str, Any]:
        """Trigger NCCL weight receive on the server."""
        return self.backend.update_weights(
            self.model_id,
            groups,
            bucket_size=bucket_size,
            strategy=strategy,
            engine_only=engine_only,
            direct_mode=direct_mode,
        )

    def close_weight_sync(self) -> dict[str, Any]:
        """Destroy persistent NCCL receiver engines on the server."""
        return self.backend.close_weight_sync(self.model_id)

    def sync_weights(
        self,
        rank: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        direct: bool = False,
    ) -> dict[str, Any]:
        """Send *weights* from training *rank* to inference targets via NCCL."""
        return self.weight_sync.sync_weights(rank, weights, direct=direct)

    # ------------------------------------------------------------------
    # Wake / Sleep
    # ------------------------------------------------------------------

    def wake_up(self) -> dict[str, Any]:
        """Resume workers from sleep."""
        return self.backend.wake_up(self.model_id)

    def sleep(self, level: int = 1) -> dict[str, Any]:
        """Release GPU memory (weights + KV cache)."""
        return self.backend.sleep(self.model_id, level=level)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Server and model status."""
        return self.backend.status()

    def weights_info(self) -> dict[str, Any]:
        """Parameter metadata for the loaded model."""
        return self.backend.weights_info(self.model_id)

    # ------------------------------------------------------------------
    # Log-probs / Policy
    # ------------------------------------------------------------------

    def compute_log_prob(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Forward-only pass returning per-token log probabilities."""
        return self.backend.compute_log_prob(batch)

    def update_policy(
        self,
        batch: dict[str, Any],
        loss_type: str = "ppo",
        loss_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward + backward + optimizer step."""
        return self.backend.update_policy(batch, loss_type=loss_type, loss_config=loss_config)

    # ------------------------------------------------------------------
    # Weights / Checkpointing
    # ------------------------------------------------------------------

    def get_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over un-sharded model parameters."""
        return self.backend.get_weights()

    def save_checkpoint(self, path: str) -> dict[str, Any]:
        """Persist model + optimizer state to *path*."""
        return self.backend.save_checkpoint(path)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """Restore model + optimizer state from *path*."""
        return self.backend.load_checkpoint(path)
