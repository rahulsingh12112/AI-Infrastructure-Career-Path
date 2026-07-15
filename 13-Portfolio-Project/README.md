# Week 13: Portfolio Project

## Project: Enterprise AI Platform on AWS

Build ONE killer project that showcases ALL your skills:

### Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    VPC (Secure)                          │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ EKS      │    │ Vector   │    │ Monitoring       │  │
│  │ + GPU    │    │ DB       │    │ (Prometheus +    │  │
│  │ + vLLM   │    │ (pgvec)  │    │  Grafana)        │  │
│  └──────────┘    └──────────┘    └──────────────────┘  │
│       ↕               ↕                                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │           RAG Pipeline + Agent System            │   │
│  └──────────────────────────────────────────────────┘   │
│       ↕                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ CI/CD    │    │ Model    │    │ Cost Dashboard   │  │
│  │ Pipeline │    │ Registry │    │                  │  │
│  └──────────┘    └──────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Checklist
- [ ] EKS cluster with GPU nodes (K8s + Networking)
- [ ] Deploy LLM with vLLM + autoscaling
- [ ] RAG pipeline (Vector DB + ingestion + retrieval)
- [ ] Agent system (MCP + tools)
- [ ] MLOps CI/CD (GitOps with ArgoCD)
- [ ] Model registry & A/B testing
- [ ] Monitoring (GPU metrics, latency, cost)
- [ ] Security (VPC, IAM, encryption, guardrails)
- [ ] Architecture diagram (draw.io)
- [ ] README with setup instructions
- [ ] Cost analysis document
- [ ] Push to GitHub as public repo
