# Week 5: LLM Deployment (Managed + Self-hosted) — Comprehensive Expert Guide

## Goal
Master LLM deployment strategies: When to use managed (Bedrock) vs self-hosted (EKS+vLLM), fine-tuning deployment, and production patterns.

## Guide Structure

| # | File | Topic | Difficulty |
|---|------|-------|-----------|
| 01 | [01-AWS-Bedrock.md](./01-AWS-Bedrock.md) | AWS Bedrock: Models, APIs, Guardrails | ⭐⭐⭐ |
| 02 | [02-Self-Hosted-EKS.md](./02-Self-Hosted-EKS.md) | Self-hosted LLMs on EKS (vLLM + GPU) | ⭐⭐⭐⭐ |
| 03 | [03-Bedrock-vs-Self-Hosted.md](./03-Bedrock-vs-Self-Hosted.md) | Decision Framework: Managed vs Self-hosted | ⭐⭐⭐⭐ |
| 04 | [04-FineTuning-Deployment.md](./04-FineTuning-Deployment.md) | LoRA & QLoRA Adapter Deployment | ⭐⭐⭐⭐ |
| 05 | [05-Blue-Green-Deployments.md](./05-Blue-Green-Deployments.md) | Blue/Green & Canary for Models | ⭐⭐⭐ |
| 06 | [06-Cost-Analysis.md](./06-Cost-Analysis.md) | Cost Analysis at Different Traffic Levels | ⭐⭐⭐⭐ |
| 07 | [07-Hands-On-Labs.md](./07-Hands-On-Labs.md) | Complete Hands-On Lab Guide | ⭐⭐⭐⭐⭐ |

## How to Use
1. Read sequentially (01 → 07)
2. Each file: Theory → Architecture → Code → Real-world → Interview Qs
3. Time needed: ~2-3 weeks (2-3 hrs/day)

## Prerequisites
- AWS Account with Bedrock access
- EKS cluster (or ability to create one)
- Understanding of vLLM/TGI (from Week 4)
- Python + boto3 + SageMaker SDK
