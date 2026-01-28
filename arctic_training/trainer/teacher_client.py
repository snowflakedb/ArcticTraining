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

"""Teacher client for On-Policy Distillation.

This module provides a client for querying a teacher model served via vLLM
to obtain log probabilities for student-generated trajectories.
"""

import asyncio
from dataclasses import dataclass
from typing import List
from typing import Optional

import aiohttp
import torch

from arctic_training.logging import logger
from arctic_training.synth.vllm_utils import kill_processes
from arctic_training.synth.vllm_utils import launch_vllm_servers


@dataclass
class TeacherLogprobResult:
    """Result from teacher logprob computation."""

    logprobs: torch.Tensor  # Shape: (batch_size, seq_len)
    """Per-token log probabilities from the teacher model."""

    mask: torch.Tensor  # Shape: (batch_size, seq_len)
    """Mask indicating valid tokens (1) vs padding (0)."""


class TeacherClient:
    """Client for querying teacher model logprobs via vLLM server.

    This client supports two modes:
    1. External server: Connect to an existing vLLM server via URL
    2. Local server: Launch a local vLLM server on specified GPUs

    The client provides methods to compute log probabilities for given
    token sequences, which is used in on-policy distillation to compute
    the reverse KL divergence loss.
    """

    def __init__(
        self,
        server_url: str = "",
        model_path: str = "",
        tensor_parallel: int = 1,
        gpu_ids: Optional[List[int]] = None,
        timeout: float = 300.0,
    ):
        """Initialize the teacher client.

        Args:
            server_url: URL of external vLLM server. If provided, uses this server.
            model_path: Path/name of model to load. Used for local server launch.
            tensor_parallel: Tensor parallelism for local server.
            gpu_ids: GPU IDs for local server. If None, auto-selects.
            timeout: Timeout in seconds for HTTP requests.
        """
        self.timeout = timeout
        self.process_ids: List[int] = []
        self._owns_server = False

        if server_url:
            # Use external server
            self.server_url = server_url.rstrip("/")
            self.model_name = ""  # Will be queried from server
            logger.info(f"TeacherClient using external vLLM server at {self.server_url}")
        elif model_path:
            # Launch local server
            if gpu_ids is None:
                gpu_ids = [0]
            self._launch_local_server(model_path, tensor_parallel, gpu_ids)
            self._owns_server = True
        else:
            raise ValueError("Either server_url or model_path must be provided")

    def _launch_local_server(
        self,
        model_path: str,
        tensor_parallel: int,
        gpu_ids: List[int],
    ) -> None:
        """Launch a local vLLM server for the teacher model."""
        logger.info(f"Launching local vLLM server for teacher model: {model_path}")
        logger.info(f"Using GPUs: {gpu_ids}, tensor_parallel: {tensor_parallel}")

        self.process_ids, urls = launch_vllm_servers(
            model_name=model_path,
            tensor_parallelism=tensor_parallel,
            gpu_ids=gpu_ids,
        )

        if not urls:
            raise RuntimeError("Failed to launch vLLM server")

        # Use the first server URL (remove the /v1/chat/completions suffix)
        base_url = urls[0].rsplit("/v1/", 1)[0]
        self.server_url = base_url
        self.model_name = model_path
        logger.info(f"Teacher vLLM server launched at {self.server_url}")

    async def _compute_logprobs_async(
        self,
        session: aiohttp.ClientSession,
        prompt_tokens: List[int],
        completion_tokens: List[int],
    ) -> List[float]:
        """Compute logprobs for a single sequence asynchronously.

        Uses the vLLM completions API with echo=True and logprobs to get
        the teacher's log probabilities for the given tokens.
        """
        # Combine prompt and completion tokens
        all_tokens = prompt_tokens + completion_tokens

        # Use the completions endpoint with prompt as token IDs
        url = f"{self.server_url}/v1/completions"

        payload = {
            "prompt": all_tokens,
            "max_tokens": 0,  # We don't want to generate, just get logprobs
            "echo": True,  # Return logprobs for prompt tokens
            "logprobs": 1,  # Return top-1 logprob (the actual token's logprob)
        }

        if self.model_name:
            payload["model"] = self.model_name

        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Teacher logprob query failed: {response.status} - {error_text}")
                    # Return zeros on error
                    return [0.0] * len(completion_tokens)

                result = await response.json()

                # Extract logprobs from response
                # vLLM returns logprobs for each token position
                choices = result.get("choices", [])
                if not choices:
                    return [0.0] * len(completion_tokens)

                logprobs_data = choices[0].get("logprobs", {})
                token_logprobs = logprobs_data.get("token_logprobs", [])

                # The first token has no logprob (it's the start), skip prompt tokens
                # We want logprobs for the completion tokens only
                prompt_len = len(prompt_tokens)
                completion_logprobs = token_logprobs[prompt_len:] if len(token_logprobs) > prompt_len else []

                # Pad or truncate to match completion length
                if len(completion_logprobs) < len(completion_tokens):
                    completion_logprobs.extend([0.0] * (len(completion_tokens) - len(completion_logprobs)))
                elif len(completion_logprobs) > len(completion_tokens):
                    completion_logprobs = completion_logprobs[: len(completion_tokens)]

                # Handle None values (first token usually has None logprob)
                completion_logprobs = [lp if lp is not None else 0.0 for lp in completion_logprobs]

                return completion_logprobs

        except asyncio.TimeoutError:
            logger.error("Teacher logprob query timed out")
            return [0.0] * len(completion_tokens)
        except Exception as e:
            logger.error(f"Teacher logprob query error: {e}")
            return [0.0] * len(completion_tokens)

    async def _batch_compute_logprobs_async(
        self,
        prompt_tokens_batch: List[List[int]],
        completion_tokens_batch: List[List[int]],
    ) -> List[List[float]]:
        """Compute logprobs for a batch of sequences asynchronously."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                self._compute_logprobs_async(session, prompt, completion)
                for prompt, completion in zip(prompt_tokens_batch, completion_tokens_batch)
            ]
            return await asyncio.gather(*tasks)

    def compute_logprobs(
        self,
        prompt_tokens: List[List[int]],
        completion_tokens: List[List[int]],
        device: torch.device,
    ) -> TeacherLogprobResult:
        """Compute teacher log probabilities for given trajectories.

        Args:
            prompt_tokens: List of prompt token sequences (batch_size,)
            completion_tokens: List of completion token sequences (batch_size,)
            device: Device to place output tensors on

        Returns:
            TeacherLogprobResult containing:
                - logprobs: Tensor of shape (batch_size, max_completion_len)
                - mask: Tensor of shape (batch_size, max_completion_len)
        """
        # Run async computation
        logprobs_list = asyncio.run(self._batch_compute_logprobs_async(prompt_tokens, completion_tokens))

        # Find max completion length for padding
        max_len = max(len(lp) for lp in logprobs_list) if logprobs_list else 0

        # Pad logprobs and create mask
        batch_size = len(logprobs_list)
        logprobs_tensor = torch.zeros(batch_size, max_len, device=device)
        mask_tensor = torch.zeros(batch_size, max_len, device=device)

        for i, lp in enumerate(logprobs_list):
            seq_len = len(lp)
            logprobs_tensor[i, :seq_len] = torch.tensor(lp, device=device)
            mask_tensor[i, :seq_len] = 1.0

        return TeacherLogprobResult(logprobs=logprobs_tensor, mask=mask_tensor)

    def compute_logprobs_from_full_sequences(
        self,
        input_ids: torch.Tensor,
        prompt_lengths: List[int],
        completion_lengths: List[int],
        device: torch.device,
    ) -> TeacherLogprobResult:
        """Compute teacher logprobs from full input_ids tensor.

        This is a convenience method that extracts prompt and completion
        tokens from a combined input_ids tensor.

        Args:
            input_ids: Full sequence tensor (batch_size, seq_len)
            prompt_lengths: Length of prompt for each sequence
            completion_lengths: Length of completion for each sequence
            device: Device to place output tensors on

        Returns:
            TeacherLogprobResult with logprobs for completion tokens
        """
        prompt_tokens = []
        completion_tokens = []

        for i in range(input_ids.size(0)):
            prompt_len = prompt_lengths[i]
            completion_len = completion_lengths[i]

            prompt = input_ids[i, :prompt_len].tolist()
            completion = input_ids[i, prompt_len : prompt_len + completion_len].tolist()

            prompt_tokens.append(prompt)
            completion_tokens.append(completion)

        return self.compute_logprobs(prompt_tokens, completion_tokens, device)

    def shutdown(self) -> None:
        """Shutdown the teacher client and any local servers."""
        if self._owns_server and self.process_ids:
            logger.info("Shutting down local teacher vLLM server")
            kill_processes(self.process_ids)
            self.process_ids = []

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()


class TeacherClientPool:
    """Pool of teacher clients for parallel logprob computation.

    This class manages multiple teacher client connections for improved
    throughput when querying logprobs from the teacher model.
    """

    def __init__(
        self,
        server_urls: List[str],
        timeout: float = 300.0,
    ):
        """Initialize the teacher client pool.

        Args:
            server_urls: List of vLLM server URLs
            timeout: Timeout in seconds for HTTP requests
        """
        self.clients = [TeacherClient(server_url=url, timeout=timeout) for url in server_urls]
        self._next_client = 0

    def get_client(self) -> TeacherClient:
        """Get the next client in round-robin fashion."""
        client = self.clients[self._next_client]
        self._next_client = (self._next_client + 1) % len(self.clients)
        return client

    def compute_logprobs(
        self,
        prompt_tokens: List[List[int]],
        completion_tokens: List[List[int]],
        device: torch.device,
    ) -> TeacherLogprobResult:
        """Compute logprobs using the next available client."""
        return self.get_client().compute_logprobs(prompt_tokens, completion_tokens, device)

    def shutdown(self) -> None:
        """Shutdown all clients in the pool."""
        for client in self.clients:
            client.shutdown()
