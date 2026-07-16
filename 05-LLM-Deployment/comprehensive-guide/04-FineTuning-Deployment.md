# 04 — LoRA & QLoRA Adapter Deployment

## What is LoRA?

Low-Rank Adaptation — fine-tune LLMs by training only small "adapter" weights instead of full model.

```
Full fine-tuning:
├── Train ALL 7B/70B parameters
├── Needs: 4-8× model size in GPU memory
├── Creates: Complete new model (same size as original)
└── Expensive, slow, inflexible

LoRA:
├── Freeze original model weights
├── Add small trainable matrices (rank-16/32) beside each layer
├── Train ONLY adapter matrices (0.1-1% of parameters)
├── Creates: Tiny adapter file (10-100 MB vs 14 GB full model)
├── Multiple adapters on same base model (multi-tenant!)
└── Cheap, fast, flexible
```

## Deploying LoRA Adapters with vLLM

```bash
# vLLM supports multiple LoRA adapters on same base model
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --enable-lora \
  --lora-modules \
    customer-support=./adapters/customer-support \
    code-gen=./adapters/code-generation \
  --max-loras 4 \
  --max-lora-rank 32
```

## Multi-Tenant Architecture

```
1 base model + N adapters = N custom models on 1 GPU!
Cost: Same as 1 model, not N models.
```

## Interview Questions

1. **Q: 100 customers, each needs custom model?**
   A: Multi-tenant LoRA — 1 base model + 100 adapters (50MB each). vLLM loads/unloads adapters on demand.

2. **Q: LoRA rank selection?**
   A: r=16 default. r=8 for simple tasks. r=32-64 for complex domain adaptation.
