"""Configuration models for the Arctic RL client."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WeightSyncConfig(BaseModel):
    """NCCL weight-transfer topology between training GPUs and inference replicas."""

    training_sharding: str = Field(
        default="dp",
        description="Training parallelism strategy: 'dp' or 'fsdp'",
    )
    training_gpus: int = 1
    inference_replicas: int = 1
    inference_tp: int = 1
    base_port: int = 29500
    bucket_size: int = 256 * 1024 * 1024  # 256 MB


class ServerLaunchConfig(BaseModel):
    """Controls whether the client spawns the ArcticInference server itself.

    When ``enabled=True``, :meth:`ArcticRLClient.launch_server` starts a
    ``uvicorn`` process serving ``arctic_inference.server.api:app`` on the
    host/port specified in :class:`InferenceServerConfig`.  The process is
    stopped automatically by :meth:`ArcticRLClient.shutdown`.

    This is the expected mode when running inside VeRL, where the client
    owns the full lifecycle of the inference engine.
    """

    enabled: bool = Field(
        default=False,
        description="Spawn the ArcticInference server on launch_server().",
    )
    app: str = Field(
        default="arctic_inference.server.api:app",
        description="Uvicorn app import string.  Override for testing.",
    )
    log_level: str = "info"
    startup_timeout: float = Field(
        default=120.0,
        description="Seconds to wait for the server to become healthy.",
    )
    health_check_interval: float = Field(
        default=2.0,
        description="Seconds between /status health-check polls.",
    )


class InferenceServerConfig(BaseModel):
    """Connection parameters for the ArcticInference server."""

    host: str = "localhost"
    port: int = 8000
    model_id: str | None = None


class TrainingServerConfig(BaseModel):
    """Connection parameters for an ArcticTraining server."""

    host: str = "localhost"
    port: int = 7000


class ArcticRLClientConfig(BaseModel):
    """Top-level configuration for :class:`ArcticRLClient`."""

    inference: InferenceServerConfig = Field(default_factory=InferenceServerConfig)
    training: TrainingServerConfig = Field(default_factory=TrainingServerConfig)
    weight_sync: WeightSyncConfig = Field(default_factory=WeightSyncConfig)
    server: ServerLaunchConfig = Field(default_factory=ServerLaunchConfig)
