"""DirectBackend -- talks to a standalone ArcticInference server.

Short-term backend: the client sends HTTP requests directly to the FastAPI
endpoints exposed by ``arctic_inference_server`` (``/init``, ``/generate``,
``/sync_weights``, ...).

Training methods are not yet implemented over this transport.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import requests
import torch

from arctic_training.rl_client.backends.base import Backend

logger = logging.getLogger(__name__)

_TRAINING_NOT_IMPL = (
    "Training is not yet supported via the direct HTTP backend."
)


class DirectBackend(Backend):
    """HTTP backend targeting a standalone ArcticInference server.

    Parameters
    ----------
    base_url : str
        Full base URL, e.g. ``"http://gpu-node:8000"``.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_model(
        self,
        model_config: dict[str, Any],
        model_id: str | None = None,
    ) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/init",
            json={"config": model_config, "model_id": model_id},
        )
        resp.raise_for_status()
        return resp.json()

    def shutdown_model(self, model_id: str) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/shutdown",
            params={"model_id": model_id},
        )
        resp.raise_for_status()
        return resp.json()

    def shutdown(self) -> dict[str, Any]:
        resp = self._session.post(f"{self.base_url}/shutdown")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        model_id: str,
        prompts: list[str] | None = None,
        *,
        prompt_token_ids: list[list[int]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        merged: list[str | list[int]] = []
        if prompts:
            merged.extend(prompts)
        if prompt_token_ids:
            merged.extend(prompt_token_ids)
        resp = self._session.post(
            f"{self.base_url}/generate",
            json={
                "model_id": model_id,
                "prompts": merged,
                "sampling_params": sampling_params or {},
            },
        )
        resp.raise_for_status()
        return resp.json()["results"]

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

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
        resp = self._session.post(
            f"{self.base_url}/sync_weights",
            json={
                "model_id": model_id,
                "groups": groups,
                "bucket_size": bucket_size,
                "strategy": strategy,
                "engine_only": engine_only,
                "direct_mode": direct_mode,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def close_weight_sync(self, model_id: str) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/close_weight_sync",
            params={"model_id": model_id},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Wake / Sleep
    # ------------------------------------------------------------------

    def wake_up(self, model_id: str) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/wake_up",
            json={"model_id": model_id},
        )
        resp.raise_for_status()
        return resp.json()

    def sleep(self, model_id: str, level: int = 1) -> dict[str, Any]:
        resp = self._session.post(
            f"{self.base_url}/sleep",
            json={"model_id": model_id, "level": level},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/status")
        resp.raise_for_status()
        return resp.json()

    def weights_info(self, model_id: str) -> dict[str, Any]:
        resp = self._session.get(
            f"{self.base_url}/weights_info",
            params={"model_id": model_id},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Training (not yet implemented)
    # ------------------------------------------------------------------

    def compute_log_prob(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(_TRAINING_NOT_IMPL)

    def update_policy(
        self,
        batch: dict[str, Any],
        loss_type: str = "ppo",
        loss_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(_TRAINING_NOT_IMPL)

    def get_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        raise NotImplementedError(_TRAINING_NOT_IMPL)

    def save_checkpoint(self, path: str) -> dict[str, Any]:
        raise NotImplementedError(_TRAINING_NOT_IMPL)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        raise NotImplementedError(_TRAINING_NOT_IMPL)
