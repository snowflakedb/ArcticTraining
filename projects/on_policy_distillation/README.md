# On-Policy Distillation

Train a student model using on-policy distillation where:
- Student generates its own trajectories
- Teacher provides per-token supervision via reverse KL divergence

This differs from traditional distillation where the teacher generates trajectories.

## Usage

The teacher model is loaded in-memory alongside the student, using DeepSpeed for efficient memory management. Both models must fit in GPU memory (with ZeRO-2/3).

```bash
arctic_training projects/on_policy_distillation/distill-qwen3.yaml
```

## Config Options

| Parameter | Description |
|-----------|-------------|
| `teacher_model` | Teacher model config (same format as `model`) |
| `disable_teacher_dropout` | Disable dropout in teacher (default: true) |
| `num_rollouts_per_prompt` | Number of student samples per prompt |
| `max_new_tokens` | Maximum generation length |
| `generation_temperature` | Student sampling temperature |
| `beta` | Reverse KL coefficient (higher = stronger teacher signal) |

## Example Config

```yaml
type: on_policy_distillation

# Student model
model:
  name_or_path: Qwen/Qwen3-1.7B
  dtype: bf16

# Teacher model (loaded in-memory)
teacher_model:
  name_or_path: Qwen/Qwen3-8B
  dtype: bf16

# Distillation settings
num_rollouts_per_prompt: 4
max_new_tokens: 1024
beta: 1.0
```

## Memory Considerations

Both student and teacher models are loaded in GPU memory. Options to reduce memory:
- Use DeepSpeed ZeRO-3 for both models
- Use lower precision for teacher (e.g., `dtype: fp16`)
- Use a smaller teacher model

## Reference

- [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)
