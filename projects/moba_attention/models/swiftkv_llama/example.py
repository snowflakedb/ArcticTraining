#!/usr/bin/env python3
# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of SwiftKV + MOBA LLaMA models.

This demonstrates the clean inheritance approach where SwiftKVMoba
inherits from SwiftKV without modifying the original implementation.
"""

import torch
from transformers import AutoTokenizer

# Import our SwiftKV + MOBA functionality
from . import (
    create_swiftkv_moba_model,
    create_swiftkv_moba_config,
    register_llama_swiftkv_moba,
    LlamaSwiftKVMoBAConfig,
    LlamaSwiftKVMoBAForCausalLM,
)


def basic_example():
    """Basic example using the convenience function."""
    print("=== SwiftKV + MOBA Basic Example ===")
    
    # This is the simplest way - one function call
    model = create_swiftkv_moba_model(
        "meta-llama/Llama-2-7b-hf",
        swiftkv=True,
        num_key_value_layers=16,     # Use SwiftKV for layers 16+
        key_value_group_size=2,      # Share KV across 2-layer groups  
        moba_chunk_size=2048,        # MOBA chunk size
        moba_topk=4,                 # MOBA top-k
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Test generation
    prompt = "The benefits of efficient attention mechanisms are"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"Input: {prompt}")
    print(f"Model type: {type(model.config).__name__}")
    print(f"SwiftKV enabled: {model.config.swiftkv}")
    print(f"Attention implementation: {model.config._attn_implementation}")
    print(f"MOBA chunk_size: {model.config.moba_chunk_size}")
    print(f"MOBA topk: {model.config.moba_topk}")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")


def inheritance_example():
    """Example showing the clean inheritance approach."""
    print("\n=== Inheritance Approach Example ===")
    
    # Register models
    register_llama_swiftkv_moba()
    
    # Create SwiftKVMoba config (inherits from SwiftKV)
    config = LlamaSwiftKVMoBAConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        # SwiftKV parameters (inherited)
        swiftkv=True,
        num_key_value_layers=16,
        key_value_group_size=1,
        # MOBA parameters (new)
        moba_chunk_size=4096,
        moba_topk=8,
        _attn_implementation="moba",
    )
    
    print(f"Config type: {type(config).__name__}")
    print(f"Inherits from: {type(config).__bases__[0].__name__}")
    print(f"Model type: {config.model_type}")
    print(f"SwiftKV params: layers={config.num_key_value_layers}, group_size={config.key_value_group_size}")
    print(f"MOBA params: chunk_size={config.moba_chunk_size}, topk={config.moba_topk}")
    
    # Create model with inherited config
    model = LlamaSwiftKVMoBAForCausalLM(config)
    print(f"Model class: {type(model).__name__}")
    print(f"Model uses SwiftKV implementation with MOBA attention: {config._attn_implementation}")


def config_only_example():
    """Example of creating just the config."""
    print("\n=== Config Creation Example ===")
    
    # Create config from pretrained
    config = create_swiftkv_moba_config(
        "meta-llama/Llama-2-7b-hf",
        swiftkv=True,
        num_key_value_layers=8,   # More aggressive SwiftKV
        key_value_group_size=4,   # Larger sharing groups
        moba_chunk_size=8192,     # Larger MOBA chunks
        moba_topk=16,             # More top-k blocks
    )
    
    print(f"Config loaded from: meta-llama/Llama-2-7b-hf")
    print(f"Config type: {type(config).__name__}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"SwiftKV config: {config.num_key_value_layers}/{config.num_hidden_layers} layers")
    print(f"MOBA config: chunk_size={config.moba_chunk_size}, topk={config.moba_topk}")


def direct_model_creation_example():
    """Example showing direct creation of SwiftKV + MOBA model."""
    print("\n=== Direct Model Creation Example ===")
    
    # Register models
    register_llama_swiftkv_moba()
    
    # Create SwiftKV + MOBA config
    config = LlamaSwiftKVMoBAConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
        swiftkv=True,
        num_key_value_layers=16,
        key_value_group_size=1,
        moba_chunk_size=4096,
        moba_topk=8,
    )
    
    # Create model directly - MOBA is built-in!
    model = LlamaSwiftKVMoBAForCausalLM(config)
    
    print(f"Model class: {type(model).__name__}")
    print(f"Config class: {type(model.config).__name__}")
    print(f"Attention implementation: {model.config._attn_implementation}")
    print(f"MOBA config: chunk_size={model.config.moba_chunk_size}, topk={model.config.moba_topk}")
    
    # Can update MOBA config dynamically
    model.update_moba_config(chunk_size=8192, topk=16)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Running examples without GPU.")
    
    # Run examples
    inheritance_example()
    config_only_example()
    direct_model_creation_example()
    
    # Only run full model example if CUDA available
    if torch.cuda.is_available():
        basic_example()
    
    print("\n=== Summary ===")
    print("Clean inheritance approach:")
    print("1. SwiftKV implementation remains unchanged")
    print("2. LlamaSwiftKVMoBAConfig inherits from LlamaSwiftKVConfig")
    print("3. LlamaSwiftKVMoBAForCausalLM inherits from LlamaSwiftKVForCausalLM")
    print("4. MOBA is built directly into the model class")
    print("5. No patching or modification needed!")
    print("6. No modifications to original SwiftKV codebase!") 