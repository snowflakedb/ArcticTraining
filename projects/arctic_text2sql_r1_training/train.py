#!/usr/bin/env python3
"""
Training entry point for GRPO Text-to-SQL
Following Arctic-Text2SQL-R1 paper

This file registers the GRPO trainer with ArcticTraining framework.
The trainer is automatically registered via RegistryMeta and can be
invoked using the arctic_training CLI.

Usage:
    arctic_training projects/arctic_text2sql_r1_training/grpo-qwen-3b.yaml
"""

from grpo_trainer import GRPOTrainer, GRPOTrainerConfig

# The trainer is automatically registered via RegistryMeta
# Just import it and it's available!

if __name__ == "__main__":
    print("=" * 60)
    print("GRPO Trainer for Text-to-SQL")
    print("=" * 60)
    print("\n✅ GRPO Trainer registered!")
    print("\nUsage:")
    print("  arctic_training projects/arctic_text2sql_r1_training/grpo-qwen-3b.yaml")
    print("\nOr from this directory:")
    print("  arctic_training grpo-qwen-3b.yaml")
    print("\n" + "=" * 60)
