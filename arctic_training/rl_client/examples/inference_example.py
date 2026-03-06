#!/usr/bin/env python3
"""Example: Using ArcticRLClient with an ArcticInference server.

Prerequisites
-------------
1. Install arctic_inference with server extras:

       cd ArcticInference-internal && pip install -e ".[server]"

2. Start the server (on a GPU node):

       arctic_inference_server --host 0.0.0.0 --port 8000

3. Run this script (on any node that can reach the server):

       python inference_example.py --host <server-host> --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging

from arctic_training.rl_client import ArcticRLClient, ArcticRLClientConfig, InferenceServerConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Arctic Inference client example")
    parser.add_argument("--host", default="localhost", help="Inference server host")
    parser.add_argument("--port", type=int, default=8000, help="Inference server port")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B", help="HuggingFace model ID or path")
    parser.add_argument("--tp", type=int, default=1, help="tensor_parallel_size for the model")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # 1. Create the client
    # ------------------------------------------------------------------ #
    config = ArcticRLClientConfig(
        inference=InferenceServerConfig(host=args.host, port=args.port),
    )
    client = ArcticRLClient(config)

    # ------------------------------------------------------------------ #
    # 2. Load a model on the server
    # ------------------------------------------------------------------ #
    log.info("Initializing model %s (tp=%d) ...", args.model, args.tp)
    init_resp = client.init_model({
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "quantization": "fp8",
        "max_model_len": 4096,
    })
    model_id = init_resp["model_id"]
    log.info("Model loaded: %s", model_id)

    # ------------------------------------------------------------------ #
    # 3. Generate text
    # ------------------------------------------------------------------ #
    prompts = [
        "Explain the theory of relativity in one paragraph.",
        "Write a Python function that computes the Fibonacci sequence.",
    ]
    log.info("Generating responses for %d prompts ...", len(prompts))

    results = client.generate(
        prompts=prompts,
        sampling_params={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 256,
        },
    )
    for i, r in enumerate(results):
        text = r.get("text", r.get("outputs", [{}])[0].get("text", ""))
        log.info("Prompt %d response:\n%s\n", i, text[:500])

    # ------------------------------------------------------------------ #
    # 4. Check server status
    # ------------------------------------------------------------------ #
    status = client.status()
    log.info("Server status:\n%s", json.dumps(status, indent=2))

    # ------------------------------------------------------------------ #
    # 5. Clean up
    # ------------------------------------------------------------------ #
    log.info("Shutting down model %s ...", model_id)
    client.shutdown_model(model_id)
    log.info("Done.")


if __name__ == "__main__":
    main()
