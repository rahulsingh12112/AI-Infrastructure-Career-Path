# Model Serving & LLM Inference — Comprehensive Expert Guide

## Goal
Master LLM inference infrastructure at expert level.
Target: Deploy, optimize, and scale large language models in production.

## Guide Structure

| # | File | Topic | Difficulty |
|---|------|-------|-----------|
| 01 | [01-vLLM.md](./01-vLLM.md) | vLLM: PagedAttention & Continuous Batching | ⭐⭐⭐⭐ |
| 02 | [02-Triton-Inference-Server.md](./02-Triton-Inference-Server.md) | NVIDIA Triton Inference Server | ⭐⭐⭐⭐ |
| 03 | [03-TGI.md](./03-TGI.md) | HuggingFace Text Generation Inference | ⭐⭐⭐ |
| 04 | [04-Quantization.md](./04-Quantization.md) | Quantization: GPTQ, AWQ, INT4/INT8 | ⭐⭐⭐⭐ |
| 05 | [05-KV-Cache.md](./05-KV-Cache.md) | KV-Cache Optimization | ⭐⭐⭐⭐⭐ |
| 06 | [06-Autoscaling.md](./06-Autoscaling.md) | Autoscaling Inference Endpoints | ⭐⭐⭐ |
| 07 | [07-Benchmarking.md](./07-Benchmarking.md) | Benchmarking: Latency, Throughput, Cost | ⭐⭐⭐⭐ |
| 08 | [08-Hands-On-Labs.md](./08-Hands-On-Labs.md) | Complete Hands-On Lab Guide | ⭐⭐⭐⭐⭐ |

## How to Use This Guide
1. Read sequentially (01 → 08)
2. Each file: Theory → Architecture → Code → Real-world → Interview Qs
3. Do hands-on labs after every 2 topics
4. Time needed: ~3-4 weeks (2-3 hrs/day)

## Prerequisites
- GPU access (at least 1× A10G or T4)
- Docker installed
- Python 3.10+ with PyTorch, transformers
- Understanding of Transformer architecture (attention mechanism)
