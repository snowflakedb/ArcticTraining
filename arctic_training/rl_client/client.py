"""ArcticRLClient -- unified client for Arctic servers.

All operations go through a single :class:`Backend` instance.  The client
caches ``model_id`` after :meth:`init_model` so callers don't need to pass
it on every call.

When ``config.server.enabled`` is ``True``, the client also owns the
ArcticInference server process: :meth:`launch_server` spawns it and
:meth:`shutdown` tears it down.
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
        self.backend = _build_backend(config)
        self.weight_sync = WeightSyncCoordinator(config.weight_sync)
        self.model_id: str | None = config.inference.model_id

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
        base_url = f"http://{self.config.inference.host}:{self.config.inference.port}"
        deadline = time.monotonic() + srv.startup_timeout

        while time.monotonic() < deadline:
            if self._server_process is not None and self._server_process.poll() is not None:
                raise RuntimeError(
                    f"Server process exited with code {self._server_process.returncode}"
                )
            try:
                resp = requests.get(f"{base_url}/status", timeout=3)
                if resp.ok:
                    logger.info("ArcticInference server is ready at %s", base_url)
                    return
            except requests.ConnectionError:
                pass
            time.sleep(srv.health_check_interval)

        raise TimeoutError(
            f"ArcticInference server did not become healthy within "
            f"{srv.startup_timeout}s at {base_url}"
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
        data = self.backend.init_model(model_config, model_id=model_id)
        self.model_id = data.get("model_id") or model_id or model_config.get("model", "default")
        data["model_id"] = self.model_id
        return data

    def shutdown_model(self, model_id: str | None = None) -> dict[str, Any]:
        """Unload a single model."""
        return self.backend.shutdown_model(model_id or self.model_id)

    def shutdown(self) -> None:
        """Tear down NCCL senders, shut down the server, and stop the process."""
        self.weight_sync.destroy()
        try:
            self.backend.shutdown()
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
