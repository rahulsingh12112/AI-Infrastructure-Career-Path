# ⏱️ 14-Week Learning Timeline

## Month 1: AI/ML Infrastructure (Your Bread & Butter)

### Week 1: GPU Infrastructure on AWS
**Why:** Your networking background + GPU cluster networking = RARE skill. Companies like NVIDIA, Google hiring for this.

**Learn:**
- [ ] EC2 GPU instances (P4d, P5, G5, Inf2)
- [ ] AWS Inferentia & Trainium chips
- [ ] EFA (Elastic Fabric Adapter) for GPU networking
- [ ] NVIDIA NCCL (multi-GPU communication)
- [ ] GPU cluster networking (your networking expertise directly applies)
- [ ] Placement groups for ML workloads
- [ ] Cost optimization (Spot instances for training)

**Hands-on:**
- [ ] Launch a multi-GPU training job
- [ ] Configure EFA for distributed training
- [ ] Compare Inferentia vs GPU performance

---

### Week 2: Kubernetes for ML
**Why:** K8s aata hai + ML scheduling = deadly combo. Cisco, Google, Salesforce hiring for this.

**Learn:**
- [ ] NVIDIA GPU Operator on K8s
- [ ] KubeFlow — ML pipelines on Kubernetes
- [ ] Ray — distributed ML framework
- [ ] GPU scheduling & resource quotas
- [ ] Node affinity for GPU nodes
- [ ] Karpenter for GPU auto-scaling
- [ ] Multi-tenancy for ML teams

**Hands-on:**
- [ ] Deploy KubeFlow on EKS
- [ ] Run distributed training job on K8s
- [ ] Set up Ray cluster for inference

---

### Week 3: SageMaker Advanced
**Why:** AWS ecosystem mein ho, SageMaker depth chahiye. Most AWS AI roles need this.

**Learn:**
- [ ] SageMaker Pipelines (ML CI/CD)
- [ ] Multi-model endpoints
- [ ] SageMaker Inference Recommender
- [ ] Training Compiler & Distributed Training
- [ ] SageMaker HyperPod (GPU cluster management)
- [ ] Model Registry & Lineage
- [ ] SageMaker + Step Functions orchestration

**Hands-on:**
- [ ] Build end-to-end ML pipeline
- [ ] Deploy multi-model endpoint
- [ ] Use HyperPod for distributed training

---

### Week 4: Model Serving & Inference
**Why:** Model deploy karna = most demanded skill. Every AI infra job needs this.

**Learn:**
- [ ] vLLM (LLM serving, PagedAttention)
- [ ] NVIDIA Triton Inference Server
- [ ] TorchServe
- [ ] Text Generation Inference (TGI by HuggingFace)
- [ ] Batching strategies (dynamic, continuous)
- [ ] Model quantization (GPTQ, AWQ, GGUF)
- [ ] KV-cache optimization
- [ ] Autoscaling inference endpoints

**Hands-on:**
- [ ] Deploy Llama model with vLLM
- [ ] Set up Triton server with multiple models
- [ ] Benchmark: latency, throughput, cost comparison

---

## Month 2: GenAI Infrastructure

### Week 5: LLM Deployment (Self-hosted + Managed)
**Why:** Companies self-host LLMs for privacy/cost. Both Bedrock + self-hosted knowledge needed.

**Learn:**
- [ ] AWS Bedrock (Claude, Titan, Llama models)
- [ ] Self-hosted LLMs on EKS (with GPU nodes)
- [ ] Bedrock vs Self-hosted: when to use which
- [ ] Fine-tuning deployment (LoRA, QLoRA adapters)
- [ ] Model artifacts management (S3, Model Registry)
- [ ] Blue/Green deployments for models
- [ ] Cost analysis: Bedrock vs Self-hosted

**Hands-on:**
- [ ] Deploy Llama-3 on EKS with vLLM
- [ ] Set up Bedrock with guardrails
- [ ] Fine-tune a model and deploy the adapter

---

### Week 6: RAG Infrastructure
**Why:** RAG infra = Vector DB deployment, scaling, monitoring. Your infra skills directly apply.

**Learn:**
- [ ] Vector DB deployment & scaling (Pinecone, pgvector on RDS, Weaviate on K8s)
- [ ] Bedrock Knowledge Bases (managed RAG)
- [ ] OpenSearch vector search (AWS native)
- [ ] Embedding model serving (batch + real-time)
- [ ] Data ingestion pipelines (S3 → chunk → embed → store)
- [ ] Caching strategies for RAG
- [ ] Monitoring vector DB performance

**Hands-on:**
- [ ] Deploy Weaviate on EKS with persistence
- [ ] Set up pgvector on Aurora PostgreSQL
- [ ] Build ingestion pipeline with Step Functions
- [ ] Benchmark: latency at different scales

---

### Week 7: RAG Application Layer
**Why:** Need to understand what you're deploying. Application knowledge makes you better infra engineer.

**Learn:**
- [ ] LangChain / LlamaIndex basics
- [ ] Embeddings (OpenAI, Cohere, HuggingFace)
- [ ] Chunking strategies
- [ ] Retrieval methods (similarity, hybrid, reranking)
- [ ] RAG evaluation (RAGAS framework)
- [ ] Production RAG patterns

**Hands-on:**
- [ ] Build RAG chatbot on company docs
- [ ] Implement hybrid search
- [ ] Evaluate with RAGAS

---

### Week 8: Agentic AI & MCP
**Why:** Hottest skill in market. 300% job growth. Even infra engineers need to understand what agents do.

**Learn:**
- [ ] Agent fundamentals (ReAct, tool use, function calling)
- [ ] Multi-agent orchestration (CrewAI, LangGraph)
- [ ] MCP (Model Context Protocol) — connecting agents to tools
- [ ] Bedrock Agents
- [ ] Agent infrastructure requirements
- [ ] Scaling agent workloads

**Hands-on:**
- [ ] Build multi-agent system
- [ ] Create MCP server
- [ ] Deploy agents on AWS with proper infra

---

## Month 3: MLOps + Production

### Week 9: MLOps (CI/CD for ML)
**Why:** Your DevOps/K8s background → MLOps is natural extension. High demand role.

**Learn:**
- [ ] ML pipeline orchestration (SageMaker Pipelines, Airflow, Prefect)
- [ ] Model versioning & registry
- [ ] A/B testing for models
- [ ] Canary deployments for ML (deployment guardrails)
- [ ] Feature stores (SageMaker Feature Store)
- [ ] Data versioning (DVC)
- [ ] GitOps for ML (ArgoCD + ML models)

**Hands-on:**
- [ ] Set up complete CI/CD for ML model
- [ ] Implement canary deployment with CloudWatch alarms
- [ ] GitOps: model deployment triggered by git push

---

### Week 10: Monitoring & Observability for AI
**Why:** Networking background → observability is natural. Critical for production AI.

**Learn:**
- [ ] Model drift detection
- [ ] SageMaker Model Monitor
- [ ] LLM observability (LangSmith, Langfuse, Arize)
- [ ] GPU utilization monitoring (DCGM, Prometheus)
- [ ] Cost monitoring & optimization
- [ ] Latency tracking (P50, P95, P99)
- [ ] Alert systems for model degradation
- [ ] Custom CloudWatch metrics for AI workloads

**Hands-on:**
- [ ] Set up Prometheus + Grafana for GPU cluster
- [ ] Implement model drift detection
- [ ] Build cost dashboard for AI workloads

---

### Week 11: Security for AI Systems
**Why:** Network security background = direct advantage. AI security is a growing concern.

**Learn:**
- [ ] VPC design for ML workloads (private endpoints, NAT)
- [ ] Model access control (IAM, resource policies)
- [ ] Prompt injection prevention
- [ ] Data encryption (at rest, in transit, in use)
- [ ] Bedrock Guardrails
- [ ] Network isolation for training jobs
- [ ] Compliance (HIPAA, PCI for AI)
- [ ] Secret management for API keys

**Hands-on:**
- [ ] Design secure VPC for ML platform
- [ ] Implement Bedrock Guardrails
- [ ] Set up private endpoints for all AI services

---

### Week 12: Complete AWS AI Stack
**Why:** Full AWS AI ecosystem knowledge = AWS job-ready.

**Learn:**
- [ ] Bedrock Agents + Knowledge Bases (end-to-end)
- [ ] SageMaker + Step Functions + EventBridge
- [ ] AWS Trainium/Inferentia optimization
- [ ] SageMaker HyperPod management
- [ ] Cost optimization strategies
- [ ] Multi-account AI platform setup (Control Tower)
- [ ] AWS Well-Architected for ML

**Hands-on:**
- [ ] Build complete AI platform architecture
- [ ] Cost optimization: reduce inference costs by 50%
- [ ] Multi-account setup for ML teams

---

## Month 3.5: Portfolio + Interview

### Week 13: Portfolio Project
**Build ONE killer project that showcases everything:**

**Project: Enterprise AI Platform on AWS**
- [ ] EKS cluster with GPU nodes (K8s + Networking)
- [ ] Deploy LLM (vLLM on EKS) with autoscaling
- [ ] RAG pipeline (Vector DB + ingestion + retrieval)
- [ ] Agent system (MCP + tools)
- [ ] MLOps (CI/CD, model registry, A/B testing)
- [ ] Monitoring (GPU metrics, latency, cost)
- [ ] Security (VPC, IAM, encryption)
- [ ] Documentation + Architecture diagram
- [ ] Push to GitHub as public repo

---

### Week 14: Interview Prep + Job Applications
- [ ] Resume rewrite (AI Infrastructure focused)
- [ ] LinkedIn profile optimization
- [ ] System design practice (ML system design)
- [ ] Apply to target companies (see README for list)
- [ ] Network with AI Infra community
- [ ] Mock interviews

---

## 📅 Daily Schedule Suggestion (Working Professional)

| Time | Activity |
|------|----------|
| 6:00 - 7:30 AM | Study (theory, videos, docs) |
| 8:00 PM - 10:00 PM | Hands-on (labs, coding, projects) |
| Weekends (4-5 hrs) | Project work + revision |

**Total: ~25 hours/week → Enough for this roadmap**
