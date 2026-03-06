"""Configuration models for the Arctic RL client."""

from __future__ import annotations

from typing import Literal

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


class InferenceServerConfig(BaseModel):
    """Connection parameters for an inference server.

    Set *backend* to ``"direct"`` (default) to talk to a standalone
    ArcticInference server, or ``"dss"`` to talk to a dss-platform
    SFTP server.  The rest of the config stays the same.
    """

    host: str = "localhost"
    port: int = 8000
    backend: Literal["direct", "dss"] = "direct"
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
