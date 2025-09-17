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

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from arctic_training.trainer.trainer import Trainer

import boto3
import torch

from arctic_training.checkpoint.hf_engine import HFCheckpointEngine
from arctic_training.config.checkpoint import CheckpointConfig
from arctic_training.logging import logger


class BiencoderS3CheckpointConfig(CheckpointConfig):
    type: str = "biencoder_s3"
    s3_path: str  # S3 path like s3://bucket/path/to/checkpoints
    local_cache_dir: Optional[str] = None  # Local cache directory
    max_local_checkpoints: int = 3  # Maximum number of checkpoints to keep locally


class BiencoderS3CheckpointEngine(HFCheckpointEngine):
    name = "biencoder_s3"
    config: BiencoderS3CheckpointConfig

    def __init__(self, trainer: "Trainer", config: BiencoderS3CheckpointConfig) -> None:
        super().__init__(trainer, config)
        self.s3_client = boto3.client('s3')
        
        # Parse S3 path
        if not config.s3_path.startswith("s3://"):
            raise ValueError(f"S3 path must start with s3://, got: {config.s3_path}")
        
        path_parts = config.s3_path[5:].split("/", 1)
        self.s3_bucket = path_parts[0]
        self.s3_prefix = path_parts[1] if len(path_parts) > 1 else ""
        
        # Setup local cache directory
        if config.local_cache_dir:
            self.local_cache_dir = Path(config.local_cache_dir)
        else:
            self.local_cache_dir = Path("/tmp") / "arctic_embed_checkpoints_cache"
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"S3 checkpoint engine initialized: bucket={self.s3_bucket}, prefix={self.s3_prefix}, "
                   f"local_cache={self.local_cache_dir}, max_local_checkpoints={config.max_local_checkpoints}")

    @property
    def biencoder_config_file(self) -> Path:
        return self.checkpoint_dir / "biencoder_config.json"

    
    @property
    def s3_checkpoint_prefix(self) -> str:
        """S3 prefix for current checkpoint"""
        return f"{self.s3_prefix}/global_step_{self.trainer.global_step}"

    def _upload_to_s3(self, local_path: Path, s3_key: str) -> None:
        """Upload a file to S3"""
        try:
            self.s3_client.upload_file(str(local_path), self.s3_bucket, s3_key)
            logger.info(f"Uploaded {local_path} to s3://{self.s3_bucket}/{s3_key}")
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to S3: {e}")
            raise

    def _download_from_s3(self, s3_key: str, local_path: Path) -> None:
        """Download a file from S3"""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
            logger.info(f"Downloaded s3://{self.s3_bucket}/{s3_key} to {local_path}")
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            raise

    def _list_s3_checkpoints(self) -> list[int]:
        """List all available checkpoints in S3 and return sorted global steps"""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=self.s3_prefix + "/")
            
            global_steps = set()
            for page in pages:
                if 'Contents' not in page:
                    continue
                for obj in page['Contents']:
                    key = obj['Key']
                    # Extract global_step from path like prefix/global_step_1000/...
                    parts = key.split('/')
                    for part in parts:
                        if part.startswith('global_step_'):
                            try:
                                step = int(part.split('_')[2])
                                global_steps.add(step)
                            except (IndexError, ValueError):
                                continue
            
            return sorted(global_steps)
        except Exception as e:
            logger.error(f"Failed to list S3 checkpoints: {e}")
            return []
    
    def _cleanup_local_cache(self) -> None:
        """Remove old checkpoints from local cache to stay within limit"""
        # List all checkpoint directories in cache
        checkpoint_dirs = []
        for item in self.local_cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("global_step_"):
                try:
                    step = int(item.name.split("_")[2])
                    checkpoint_dirs.append((step, item))
                except (IndexError, ValueError):
                    continue
        
        # Sort by step number (oldest first)
        checkpoint_dirs.sort(key=lambda x: x[0])
        
        # Remove oldest checkpoints if we exceed the limit
        while len(checkpoint_dirs) >= self.config.max_local_checkpoints:
            step, dir_to_remove = checkpoint_dirs.pop(0)
            logger.info(f"Removing old checkpoint from cache: {dir_to_remove}")
            shutil.rmtree(dir_to_remove, ignore_errors=True)

    def save(self, model) -> None:
        """Save model checkpoint locally then upload to S3
        
        For multi-node training:
        - All ranks save model weights locally (handled by parent class)
        - Only rank 0 uploads to S3
        """
        # Clean up old local checkpoints before saving new one
        self._cleanup_local_cache()
        
        # The model is already a DeepSpeedEngine, use DeepSpeed's checkpoint saving
        # which includes optimizer & scheduler states
        model.save_checkpoint(
            self.checkpoint_dir,
            tag=f"global_step_{self.trainer.global_step}",
            client_state={
                "train_batch_idx": self.trainer.train_batch_idx,
                "epoch_idx": self.trainer.epoch_idx,
            }
        )
        
        # Only rank 0 saves additional metadata files
        if self.global_rank == 0:
            # Save biencoder configuration
            # Access the underlying Biencoder model from DeepSpeedEngine
            biencoder_model = model.module if hasattr(model, 'module') else model
            biencoder_config = {"pooling": biencoder_model.pooling}
            self.biencoder_config_file.write_text(json.dumps(biencoder_config, indent=2))
            
            # Note: Most state is saved in DeepSpeed checkpoint
            # We only save additional metadata here
            logger.info(f"DeepSpeed checkpoint saved with optimizer and scheduler states")
        
        # Synchronize all ranks before uploading
        if self.trainer.world_size > 1:
            torch.distributed.barrier()
        
        # Only rank 0 uploads to S3
        if self.global_rank == 0:
            for local_file in self.checkpoint_dir.rglob("*"):
                if local_file.is_file():
                    relative_path = local_file.relative_to(self.checkpoint_dir)
                    s3_key = f"{self.s3_checkpoint_prefix}/{relative_path}"
                    self._upload_to_s3(local_file, s3_key)
            
            # Create a latest checkpoint marker
            latest_marker = self.local_cache_dir / "latest"
            latest_marker.write_text(str(self.trainer.global_step))
            self._upload_to_s3(latest_marker, f"{self.s3_prefix}/latest")
            
            logger.info(f"Checkpoint saved to S3 at global_step={self.trainer.global_step}")
        
        # Synchronize all ranks after upload completes
        if self.trainer.world_size > 1:
            torch.distributed.barrier()

    def load(self, model) -> None:
        """Load checkpoint from S3 for resuming training
        
        For multi-node training:
        - Only rank 0 downloads from S3
        - Checkpoint path is broadcast to all ranks
        - All ranks load from the same local checkpoint
        """
        if not self.config.auto_resume:
            logger.info("Auto-resume disabled, skipping checkpoint loading")
            return
        
        local_checkpoint_dir = None
        
        # Only rank 0 finds and downloads checkpoint
        if self.global_rank == 0:
            # Find latest checkpoint
            global_steps = self._list_s3_checkpoints()
            if not global_steps:
                logger.info("No checkpoints found in S3, starting from scratch")
            else:
                latest_step = global_steps[-1]
                logger.info(f"Found {len(global_steps)} checkpoints, loading latest: global_step_{latest_step}")
                
                # Check if checkpoint already exists in local cache
                local_checkpoint_dir = self.local_cache_dir / f"global_step_{latest_step}"
                
                if not local_checkpoint_dir.exists():
                    # Download checkpoint files
                    checkpoint_s3_prefix = f"{self.s3_prefix}/global_step_{latest_step}"
                    local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    
                    # List and download all files for this checkpoint
                    paginator = self.s3_client.get_paginator('list_objects_v2')
                    pages = paginator.paginate(Bucket=self.s3_bucket, Prefix=checkpoint_s3_prefix)
                    
                    for page in pages:
                        if 'Contents' not in page:
                            continue
                        for obj in page['Contents']:
                            s3_key = obj['Key']
                            relative_path = s3_key[len(checkpoint_s3_prefix)+1:]  # Remove prefix
                            local_path = local_checkpoint_dir / relative_path
                            self._download_from_s3(s3_key, local_path)
                    
                    logger.info(f"Downloaded checkpoint to {local_checkpoint_dir}")
                else:
                    logger.info(f"Using cached checkpoint from {local_checkpoint_dir}")
        
        # Broadcast checkpoint directory path to all ranks
        if self.trainer.world_size > 1:
            checkpoint_dir_str = str(local_checkpoint_dir) if local_checkpoint_dir else ""
            checkpoint_dir_list = [checkpoint_dir_str] if self.global_rank == 0 else [""]
            torch.distributed.broadcast_object_list(checkpoint_dir_list, src=0)
            checkpoint_dir_str = checkpoint_dir_list[0]
            
            if checkpoint_dir_str:
                local_checkpoint_dir = Path(checkpoint_dir_str)
            else:
                return  # No checkpoint to load
        elif local_checkpoint_dir is None:
            return  # No checkpoint to load
        
        # All ranks load from the same local checkpoint using DeepSpeed
        # This includes model weights, optimizer state, and scheduler state
        _, client_states = model.load_checkpoint(
            local_checkpoint_dir,
            tag=local_checkpoint_dir.name,
        )
        
        # Restore training state from client_states
        self.trainer.train_batch_idx = client_states.get("train_batch_idx", 0)
        self.trainer.epoch_idx = client_states.get("epoch_idx", 0)
        self.trainer.global_step = model.global_steps  # DeepSpeed tracks this
        
        logger.info(f"Loaded DeepSpeed checkpoint from {local_checkpoint_dir}: "
                   f"global_step={self.trainer.global_step}, "
                   f"epoch_idx={self.trainer.epoch_idx}, "
                   f"train_batch_idx={self.trainer.train_batch_idx}")
        
        # Load biencoder config
        biencoder_config_path = local_checkpoint_dir / "biencoder_config.json"
        if biencoder_config_path.exists():
            biencoder_config = json.loads(biencoder_config_path.read_text())
            # Access the underlying Biencoder model from DeepSpeedEngine
            biencoder_model = model.module if hasattr(model, 'module') else model
            biencoder_model.pooling = biencoder_config.get("pooling", biencoder_model.pooling)
        
        # Recreate dataloader to skip batches if needed
        # Only if we're in the same epoch (for your case with 1 epoch)
        if self.trainer.epoch_idx == 0 and self.trainer.train_batch_idx > 0:
            # Call the trainer's method to recreate dataloader
            if hasattr(self.trainer, '_recreate_dataloader_for_resume'):
                self.trainer._recreate_dataloader_for_resume(self.trainer.train_batch_idx)
        
        logger.info(f"Successfully resumed training with scheduler state intact")
