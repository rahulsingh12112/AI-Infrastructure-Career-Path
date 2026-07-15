# Week 6: RAG Infrastructure

## Why This Matters
RAG is #1 enterprise AI use case. The INFRA side (deploying, scaling vector DBs) is YOUR domain.

## Topics

### Vector DB Deployment Options on AWS
| Option | Type | Best For |
|--------|------|----------|
| Bedrock Knowledge Bases | Managed | Quick start, fully managed |
| OpenSearch Serverless | Managed | AWS native, serverless |
| pgvector on Aurora | Self-managed | Already using PostgreSQL |
| Pinecone | SaaS | Simplest, fastest |
| Weaviate on EKS | Self-hosted | Full control, hybrid search |

### Scaling & Operations
- Sharding strategies (by tenant, by collection)
- Replication for read scaling
- Index optimization (HNSW parameters)
- Connection pooling
- Backup & recovery

### Data Ingestion Pipeline
```
S3 (docs) → Lambda/Step Functions → Chunking → Embedding API → Vector DB
```

### Embedding Model Serving
- Bedrock Titan Embeddings (managed)
- Self-hosted on SageMaker (HuggingFace models)
- Batch embedding for bulk ingestion

### Caching
- Semantic cache (similar queries)
- Exact match cache (Redis/ElastiCache)

## Hands-on Tasks
- [ ] Deploy Weaviate on EKS with persistent storage
- [ ] Set up pgvector on Aurora PostgreSQL
- [ ] Build ingestion pipeline (S3 → chunk → embed → store)
- [ ] Set up Bedrock Knowledge Base
- [ ] Benchmark: latency at 1K, 10K, 100K, 1M vectors
- [ ] Implement semantic caching

## Resources
- [Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)
- [pgvector](https://github.com/pgvector/pgvector)
- [Weaviate on K8s](https://weaviate.io/developers/weaviate/installation/kubernetes)
- [OpenSearch vector search](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/knn.html)
