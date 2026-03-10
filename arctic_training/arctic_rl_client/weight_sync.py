"""WeightSyncCoordinator -- shared NCCL topology for training-to-inference weight transfer.

This is the single source of truth for sender/receiver IP mapping, GPU IDs,
and NCCL ports.  Both training and inference clients reference the same
coordinator instance so they can coordinate weight sync without duplicating
topology state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable

import torch

from arctic_inference.server.weight_sync.schedule import TransferSchedule
from arctic_inference.server.weight_sync.sender import WeightSender

from arctic_training.arctic_rl_client.config import WeightSyncConfig

if TYPE_CHECKING:
    from arctic_training.arctic_rl_client.client import ArcticRLClient

logger = logging.getLogger(__name__)


class WeightSyncCoordinator:
    """Manages the NCCL weight-transfer topology between training GPUs and
    inference replicas.

    Reuses :class:`TransferSchedule` for static assignment of TP workers to
    sender GPUs and :class:`WeightSender` for the actual NCCL data path.

    Parameters
    ----------
    config : WeightSyncConfig
        Topology and connection parameters.
    """

    def __init__(self, config: WeightSyncConfig) -> None:
        self.config = config
        self.schedule = TransferSchedule.build(
            training_sharding=config.training_sharding,
            training_gpus=config.training_gpus,
            inference_replicas=config.inference_replicas,
            inference_tp=config.inference_tp,
        )
        self._senders: dict[int, WeightSender] = {}

    # ------------------------------------------------------------------
    # Topology queries
    # ------------------------------------------------------------------

    @property
    def sender_ranks(self) -> list[int]:
        """Training GPU ranks that are actively sending."""
        return self.schedule.active_sender_ranks

    @property
    def num_groups(self) -> int:
        return self.schedule.num_groups

    # ------------------------------------------------------------------
    # Sender management
    # ------------------------------------------------------------------

    def get_or_create_sender(
        self,
        rank: int,
        master_addr: str,
        device: torch.device,
    ) -> WeightSender:
        """Get or lazily create a :class:`WeightSender` for *rank*."""
        if rank not in self._senders:
            group = self.schedule.groups[rank]
            self._senders[rank] = WeightSender(
                group=group,
                schedule=self.schedule,
                master_addr=master_addr,
                base_port=self.config.base_port,
                device=device,
                bucket_size=self.config.bucket_size,
            )
        return self._senders[rank]

    def sync_weights(
        self,
        rank: int,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        direct: bool = False,
    ) -> dict[str, Any]:
        """Send *weights* from training *rank* to its assigned inference targets.

        The sender must have been created beforehand via
        :meth:`get_or_create_sender`.
        """
        sender = self._senders[rank]
        return sender.send(weights, direct=direct)

    # ------------------------------------------------------------------
    # Inference-side NCCL setup
    # ------------------------------------------------------------------

    def prepare_inference_server(
        self,
        client: ArcticRLClient,
        master_addr: str,
    ) -> dict[str, Any]:
        """Tell the inference server to create NCCL receiver engines.

        Calls ``/sync_weights`` with ``engine_only=True`` so the server
        performs the NCCL rendezvous without actually receiving data.
        """
        groups = [
            {
                "group_id": g.group_id,
                "master_addr": master_addr,
                "master_port": self.config.base_port,
                "world_size": g.world_size,
                "replica_ids": g.replica_ids,
            }
            for g in self.schedule.groups
        ]
        return client.update_weights(groups, engine_only=True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Destroy all NCCL sender connections."""
        for sender in self._senders.values():
            try:
                sender.destroy()
            except Exception:
                logger.warning("Failed to destroy sender", exc_info=True)
        self._senders.clear()
