"""ArcticRLClient -- unified client for Arctic servers.

Sends HTTP requests directly to the FastAPI endpoints exposed by
``arctic_inference_server`` (``/init``, ``/generate``, ``/sync_weights``, ...).

When ``config.server.enabled`` is ``True``, the client also owns the
ArcticInference server process: :meth:`launch_server` spawns it and
:meth:`shutdown` tears it down.

Training methods are not yet implemented.
"""

from __future__ import annotations

import atexit
import logging
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Iterator

import requests
import torch

from arctic_training.arctic_rl_client.config import ArcticRLClientConfig
from arctic_training.arctic_rl_client.weight_sync import WeightSyncCoordinator

logger = logging.getLogger(__name__)

_TRAINING_NOT_IMPL = (
    "Training is not yet supported via the HTTP backend."
)


class ArcticRLClient:
    """Framework-agnostic HTTP client for Arctic servers.

    Parameters
    ----------
    config : ArcticRLClientConfig
        Connection and topology parameters.  When
        ``config.server.enabled`` is *True*, call :meth:`launch_server`
        (or pass *auto_launch=True*) to spawn the ArcticInference process
        before any other operation.
    auto_launch : bool
        If *True* **and** ``config.server.enabled`` is *True*, the server
        is started in ``__init__``.  Default is *False* so that the caller
        controls the timing.
    """

    def __init__(
        self,
        config: ArcticRLClientConfig,
        *,
        auto_launch: bool = False,
    ) -> None:
        self.config = config
        self.weight_sync = WeightSyncCoordinator(config.weight_sync)
        self.model_id: str | None = config.inference.model_id

        base_url = f"http://{config.inference.host}:{config.inference.port}"
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._server_process: subprocess.Popen | None = None

        if auto_launch and config.server.enabled:
            self.launch_server()

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def launch_server(self) -> None:
        """Spawn the ArcticInference server as a child process.

        Blocks until the server responds to a health check or
        ``config.server.startup_timeout`` is exceeded.

        Raises
        ------
        RuntimeError
            If the server is already running or fails to become healthy.
        """
        if self._server_process is not None and self._server_process.poll() is None:
            raise RuntimeError("Server process is already running")

        srv = self.config.server
        host = self.config.inference.host
        port = self.config.inference.port

        cmd = [
            sys.executable, "-m", "uvicorn",
            srv.app,
            "--host", host,
            "--port", str(port),
            "--log-level", srv.log_level,
        ]
        logger.info("Launching ArcticInference server: %s", " ".join(cmd))

        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        atexit.register(self._kill_server)

        try:
            self._wait_for_ready()
        except Exception:
            self.stop_server()
            raise

    def _wait_for_ready(self) -> None:
        """Poll the server until it responds or timeout."""
        srv = self.config.server
        deadline = time.monotonic() + srv.startup_timeout

        while time.monotonic() < deadline:
            if self._server_process is not None and self._server_process.poll() is not None:
                raise RuntimeError(
                    f"Server process exited with code {self._server_process.returncode}"
                )
            try:
                resp = requests.get(f"{self._base_url}/status", timeout=3)
                if resp.ok:
                    logger.info("ArcticInference server is ready at %s", self._base_url)
                    return
            except requests.ConnectionError:
                pass
            time.sleep(srv.health_check_interval)

        raise TimeoutError(
            f"ArcticInference server did not become healthy within "
            f"{srv.startup_timeout}s at {self._base_url}"
        )

    def stop_server(self) -> None:
        """Terminate the server process if it was launched by this client."""
        proc = self._server_process
        if proc is None:
            return
        self._server_process = None

        if proc.poll() is not None:
            return

        logger.info("Stopping ArcticInference server (pid=%d)", proc.pid)
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not exit after SIGTERM, sending SIGKILL")
            proc.kill()
            proc.wait(timeout=5)

    def _kill_server(self) -> None:
        """atexit callback -- best-effort cleanup."""
        try:
            self.stop_server()
        except Exception:
            pass

    @property
    def server_running(self) -> bool:
        """Whether a client-managed server process is alive."""
        return self._server_process is not None and self._server_process.poll() is None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def init_model(
        self,
        model_config: dict[str, Any],
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Load a model on the server.  Caches the returned ``model_id``."""
        resp = self._session.post(
            f"{self._base_url}/init",
            json={"config": model_config, "model_id": model_id},
        )
        resp.raise_for_status()
        data = resp.json()
        self.model_id = data.get("model_id") or model_id or model_config.get("model", "default")
        data["model_id"] = self.model_id
        return data

    def shutdown_model(self, model_id: str | None = None) -> dict[str, Any]:
        """Unload a single model."""
        resp = self._session.post(
            f"{self._base_url}/shutdown",
            params={"model_id": model_id or self.model_id},
        )
        resp.raise_for_status()
        return resp.json()

    def shutdown(self) -> None:
        """Tear down NCCL senders, shut down the server, and stop the process."""
        self.weight_sync.destroy()
        try:
            self._session.post(f"{self._base_url}/shutdown").raise_for_status()
        except Exception:
            logger.warning("Server shutdown request failed", exc_info=True)
        self.stop_server()

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
        merged: list[str | list[int]] = []
        if prompts:
            merged.extend(prompts)
        if prompt_token_ids:
            merged.extend(prompt_token_ids)
        resp = self._session.post(
            f"{self._base_url}/generate",
            json={
                "model_id": self.model_id,
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
        groups: list[dict[str, Any]],
        *,
        bucket_size: int = 256 * 1024 * 1024,
        strategy: str = "hotswap",
        engine_only: bool = False,
        direct_mode: bool = False,
    ) -> dict[str, Any]:
        """Trigger NCCL weight receive on the server."""
        resp = self._session.post(
            f"{self._base_url}/sync_weights",
            json={
                "model_id": self.model_id,
                "groups": groups,
                "bucket_size": bucket_size,
                "strategy": strategy,
                "engine_only": engine_only,
                "direct_mode": direct_mode,
            },
        )
        resp.raise_for_status()
        return resp.json()

    def close_weight_sync(self) -> dict[str, Any]:
        """Destroy persistent NCCL receiver engines on the server."""
        resp = self._session.post(
            f"{self._base_url}/close_weight_sync",
            params={"model_id": self.model_id},
        )
        resp.raise_for_status()
        return resp.json()

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
        resp = self._session.post(
            f"{self._base_url}/wake_up",
            json={"model_id": self.model_id},
        )
        resp.raise_for_status()
        return resp.json()

    def sleep(self, level: int = 1) -> dict[str, Any]:
        """Release GPU memory (weights + KV cache)."""
        resp = self._session.post(
            f"{self._base_url}/sleep",
            json={"model_id": self.model_id, "level": level},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Server and model status."""
        resp = self._session.get(f"{self._base_url}/status")
        resp.raise_for_status()
        return resp.json()

    def weights_info(self) -> dict[str, Any]:
        """Parameter metadata for the loaded model."""
        resp = self._session.get(
            f"{self._base_url}/weights_info",
            params={"model_id": self.model_id},
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Log-probs / Policy (not yet implemented)
    # ------------------------------------------------------------------

    def compute_log_prob(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Forward-only pass returning per-token log probabilities."""
        raise NotImplementedError(_TRAINING_NOT_IMPL)

    def update_policy(
        self,
        batch: dict[str, Any],
        loss_type: str = "ppo",
        loss_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Forward + backward + optimizer step."""
        raise NotImplementedError(_TRAINING_NOT_IMPL)

    # ------------------------------------------------------------------
    # Weights / Checkpointing (not yet implemented)
    # ------------------------------------------------------------------

    def get_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate over un-sharded model parameters."""
        raise NotImplementedError(_TRAINING_NOT_IMPL)

    def save_checkpoint(self, path: str) -> dict[str, Any]:
        """Persist model + optimizer state to *path*."""
        raise NotImplementedError(_TRAINING_NOT_IMPL)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """Restore model + optimizer state from *path*."""
        raise NotImplementedError(_TRAINING_NOT_IMPL)
