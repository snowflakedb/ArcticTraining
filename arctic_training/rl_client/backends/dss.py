"""DSSBackend -- talks to a dss-platform SFTP server.

This is the long-term backend: the client sends HTTP requests to the
SFTP server's job-based endpoints (``/initialize``, ``/generate``,
``/log-probs``, ``/destroy``, ...).  The SFTP server's
``SampleJobManager`` delegates to an ``arctic_inference.server.Driver``
internally (see ``mwyatt/new-sample-mgr`` branch).

Protocol translation handled here:
    - ``model_id`` is mapped to/from ``job_id`` transparently.
    - ``init_model(model_config)`` is translated to ``/initialize``
      with a ``JobConfig`` wrapper.
    - ``generate()`` is translated to ``/generate`` with ``job_id``.
    - Weight-sync, sleep/wake are forwarded when dss-platform exposes
      them; until then they raise ``NotImplementedError``.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Iterator

import requests
import torch

from arctic_training.rl_client.backends.base import Backend

logger = logging.getLogger(__name__)

_TRAINING_NOT_IMPL = (
    "Training is not yet supported via the DSS backend."
)


class DSSBackend(Backend):
    """HTTP backend targeting a dss-platform SFTP server.

    Parameters
    ----------
    base_url : str
        Full base URL of the SFTP server, e.g. ``"http://dss-node:7000"``.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._model_to_job: dict[str, int] = {}
        self._job_to_model: dict[int, str] = {}

    def _job_id_for(self, model_id: str) -> int:
        jid = self._model_to_job.get(model_id)
        if jid is None:
            raise RuntimeError(
                f"No DSS job mapped to model_id={model_id!r}. "
                "Call init_model() first."
            )
        return jid

    # ------------------------------------------------------------------
    # Inference: Lifecycle
    # ------------------------------------------------------------------

    def init_model(
        self,
        model_config: dict[str, Any],
        model_id: str | None = None,
    ) -> dict[str, Any]:
        job_config = {
            "model_name": model_config.get("model", "unknown"),
            "job_type": "sampling",
            "vllm_config": model_config,
        }
        resp = self._session.post(
            f"{self.base_url}/initialize",
            json=job_config,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id: int = data["job_id"]

        effective_model_id = model_id or str(job_id)
        self._model_to_job[effective_model_id] = job_id
        self._job_to_model[job_id] = effective_model_id

        logger.info("DSS job %d initialised as model_id=%s", job_id, effective_model_id)
        return {"model_id": effective_model_id, "job_id": job_id, **data}

    def shutdown_model(self, model_id: str) -> dict[str, Any]:
        job_id = self._job_id_for(model_id)
        resp = self._session.post(
            f"{self.base_url}/destroy",
            params={"job_id": job_id},
            json={"job_type": "sampling"},
        )
        resp.raise_for_status()
        self._model_to_job.pop(model_id, None)
        self._job_to_model.pop(job_id, None)
        return resp.json()

    def shutdown(self) -> dict[str, Any]:
        for model_id in list(self._model_to_job):
            try:
                self.shutdown_model(model_id)
            except Exception:
                logger.warning("Failed to destroy DSS job for %s", model_id, exc_info=True)
        return {"status": "shutdown"}

    # ------------------------------------------------------------------
    # Inference: Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        model_id: str,
        prompts: list[str] | None = None,
        *,
        prompt_token_ids: list[list[int]] | None = None,
        sampling_params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        job_id = self._job_id_for(model_id)

        if prompt_token_ids is not None:
            raise NotImplementedError(
                "DSSBackend does not yet support prompt_token_ids; "
                "pass prompts as strings."
            )

        resp = self._session.post(
            f"{self.base_url}/generate",
            params={"job_id": job_id},
            json={
                "prompts": prompts or [],
                "sampling_params": sampling_params or {},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", data.get("result", []))

    # ------------------------------------------------------------------
    # Log-probs (DSS has a dedicated /log-probs endpoint)
    # ------------------------------------------------------------------

    def log_probs(
        self,
        model_id: str,
        prompts: list[str],
        top_k: int = 1,
    ) -> list[dict[str, Any]]:
        """DSS-native log-prob endpoint (binary torch response)."""
        job_id = self._job_id_for(model_id)
        resp = self._session.post(
            f"{self.base_url}/log-probs",
            params={"job_id": job_id},
            json={"prompts": prompts, "completions": None, "top_k": top_k},
        )
        resp.raise_for_status()
        data = torch.load(io.BytesIO(resp.content), map_location="cpu")
        return data["results"]

    # ------------------------------------------------------------------
    # Inference: Weight sync (not yet exposed by dss-platform)
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
        raise NotImplementedError(
            "Weight sync is not yet exposed by dss-platform. "
            "Pending SFTP server endpoint addition."
        )

    def close_weight_sync(self, model_id: str) -> dict[str, Any]:
        raise NotImplementedError(
            "Weight sync is not yet exposed by dss-platform."
        )

    # ------------------------------------------------------------------
    # Inference: Wake / Sleep (not yet exposed by dss-platform)
    # ------------------------------------------------------------------

    def wake_up(self, model_id: str) -> dict[str, Any]:
        raise NotImplementedError(
            "Sleep/wake is not yet exposed by dss-platform."
        )

    def sleep(self, model_id: str, level: int = 1) -> dict[str, Any]:
        raise NotImplementedError(
            "Sleep/wake is not yet exposed by dss-platform."
        )

    # ------------------------------------------------------------------
    # Inference: Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/status")
        resp.raise_for_status()
        return resp.json()

    def weights_info(self, model_id: str) -> dict[str, Any]:
        raise NotImplementedError(
            "weights_info is not yet exposed by dss-platform."
        )

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
