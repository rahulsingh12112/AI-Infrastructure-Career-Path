# 02 — Scaling & Operations (Sharding, Replication, HNSW, Backup)

## Vector DB Scaling Challenges

```
Problem at scale:
├── 100M vectors × 1536 dimensions × 4 bytes = 600 GB just for vectors!
├── HNSW index adds 2-3x overhead = 1.8 TB total
├── Single node can't hold this in memory
├── Read latency must stay < 50ms at 1000 QPS
└── Need: Sharding + Replication + Index tuning
```

## Sharding Strategies

### By Collection/Namespace (Multi-tenant)
```
Tenant A: Shard 0 (Node 1) — 500K vectors
Tenant B: Shard 1 (Node 2) — 2M vectors
Tenant C: Shard 2 (Node 3) — 100K vectors

Pros: Isolation, easy to scale per tenant, easy deletion
Cons: Uneven load if tenant sizes vary
Best for: SaaS with multiple customers
```

### By Hash (Even distribution)
```
vector_id % num_shards = target_shard

All vectors distributed evenly across N shards.
Query: Fan-out to ALL shards → merge results

Pros: Even load distribution
Cons: Every query hits every shard (more network)
Best for: Single large collection, uniform access
```

### By Range (Geographic/Temporal)
```
Shard by document date:
├── Shard 0: Documents 2024-01 to 2024-06
├── Shard 1: Documents 2024-07 to 2024-12
└── Shard 2: Documents 2025-01 onwards

Query with date filter → only hit relevant shard
Pros: Efficient filtered queries
Cons: Hot shard for recent data
Best for: Time-series documents, recent data more queried
```

### Weaviate Sharding Config
```yaml
# Weaviate class with sharding
{
  "class": "Document",
  "shardingConfig": {
    "desiredCount": 3,              # Number of shards
    "virtualPerPhysical": 128,      # Virtual shards per physical
    "desiredVirtualCount": 384,
    "replicas": 2,                  # Replication factor
  },
  "replicationConfig": {
    "factor": 2,                    # 2 copies of each shard
  }
}
```

## Replication for Read Scaling

```
Architecture:
┌──────────────────────────────────────────────────────┐
│  Write path: Client → Primary → Replicas (async)     │
│  Read path:  Client → Any replica (load balanced)    │
│                                                       │
│  Primary       Replica 1     Replica 2               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Shard 0  │ │ Shard 0  │ │ Shard 0  │            │
│  │ (write)  │ │ (read)   │ │ (read)   │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│                                                       │
│  Throughput: 3x read capacity (3 replicas)           │
│  Availability: Survives 2 node failures              │
└──────────────────────────────────────────────────────┘

OpenSearch replication:
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 2
  }
}

pgvector replication:
├── Aurora: Built-in read replicas (up to 15)
├── Read queries routed to replicas
└── Writes go to primary → replicate async
```

## HNSW Index Optimization

### What is HNSW?
```
HNSW (Hierarchical Navigable Small World) = most common ANN index

Structure: Multi-layer graph
├── Layer 3 (few nodes): Long-range connections (highway)
├── Layer 2: Medium connections
├── Layer 1: More connections
└── Layer 0 (all nodes): Dense local connections

Search: Start at top layer → navigate down → refine at bottom
Like: Airplane (top layer) → highway → local streets → destination
```

### Key Parameters

```
Parameter        │ Effect on Search   │ Effect on Build  │ Effect on Memory
─────────────────┼────────────────────┼──────────────────┼─────────────────
M (connections)  │ Higher = better    │ Higher = slower  │ Higher = more
                 │ recall             │ build time       │ memory
ef_construction  │ No effect          │ Higher = better  │ No effect
                 │ (build only)       │ index quality    │
ef_search        │ Higher = better    │ No effect        │ No effect
                 │ recall, slower     │                  │

Recommended values:
├── Small dataset (<100K): M=16, ef_construction=128, ef_search=64
├── Medium (100K-10M):     M=16, ef_construction=256, ef_search=128
├── Large (10M-100M):      M=32, ef_construction=512, ef_search=256
└── Massive (>100M):       M=48, ef_construction=512, ef_search=512

Trade-off:
├── Higher recall (more accurate) = slower search = more memory
├── Production sweet spot: 95-98% recall at <10ms latency
└── Tune ef_search at query time (not rebuild needed!)
```

### OpenSearch HNSW Tuning
```json
{
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 1536,
        "method": {
          "name": "hnsw",
          "engine": "nmslib",
          "space_type": "cosinesimil",
          "parameters": {
            "m": 24,
            "ef_construction": 512
          }
        }
      }
    }
  },
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 256
    }
  }
}
```

### pgvector Index Tuning
```sql
-- HNSW index with custom params
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 24, ef_construction = 256);

-- Set search parameters (per session)
SET hnsw.ef_search = 200;

-- IVFFlat alternative (faster build, lower recall)
CREATE INDEX ON documents
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 1000);    -- sqrt(num_vectors) is good starting point

-- Set IVFFlat probe count
SET ivfflat.probes = 50;  -- Higher = better recall, slower
```

## Connection Pooling

```
Problem: Vector DB connections are expensive
├── Each connection holds state, memory
├── LLM app creates 100s of concurrent requests
├── Without pooling: 100s of connections = DB overloaded
└── With pooling: 10-20 connections shared among 100s of requests

For pgvector (PgBouncer):
┌───────────────────────────────────────────┐
│  App (100 concurrent requests)             │
│       │                                    │
│       ▼                                    │
│  PgBouncer (connection pooler)             │
│  ├── Pool size: 20 connections             │
│  ├── Mode: transaction                     │
│  └── Max client connections: 1000          │
│       │                                    │
│       ▼                                    │
│  Aurora PostgreSQL (20 actual connections)  │
└───────────────────────────────────────────┘

# PgBouncer config
[pgbouncer]
pool_mode = transaction
default_pool_size = 20
max_client_conn = 1000
```

## Backup & Recovery

```
Backup Strategy:

pgvector on Aurora:
├── Automated backups (AWS managed, daily)
├── Point-in-time recovery (up to 35 days)
├── Cross-region replication for DR
└── Aurora Backtrack (rewind without restore)

OpenSearch Serverless:
├── Automated snapshots (managed by AWS)
├── No manual backup needed
└── Data replicated across AZs

Weaviate on EKS:
├── PVC snapshots (EBS snapshots via VolumeSnapshot)
├── Weaviate backup API → S3
├── Schedule: CronJob every 6 hours
└── Recovery: Restore PVC from snapshot or import from S3

# Weaviate backup to S3
curl -X POST http://weaviate:8080/v1/backups/s3 \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "backup-20240716",
    "include": ["Document"],
    "backend": "s3",
    "config": {"bucket": "weaviate-backups", "path": "daily/"}
  }'
```

## Performance Benchmarks (Reference)

```
pgvector (Aurora r6g.2xlarge):
Vectors   │ Insert/s │ Search (ms) │ Recall@10 │ Memory
──────────┼──────────┼─────────────┼───────────┼────────
100K      │ 5,000    │ 2ms         │ 99%       │ 2 GB
1M        │ 3,000    │ 5ms         │ 98%       │ 15 GB
5M        │ 1,500    │ 15ms        │ 95%       │ 70 GB
10M       │ 800      │ 35ms        │ 92%       │ 140 GB  ← degrades!

OpenSearch Serverless:
Vectors   │ Search (ms) │ Recall@10 │ Notes
──────────┼─────────────┼───────────┼──────────────
1M        │ 8ms         │ 98%       │ Auto-scales
10M       │ 15ms        │ 97%       │ Auto-scales
100M      │ 25ms        │ 96%       │ Needs tuning

Weaviate (3-node cluster, r6i.2xlarge):
Vectors   │ Search (ms) │ Recall@10 │ QPS
──────────┼─────────────┼───────────┼──────
1M        │ 3ms         │ 99%       │ 5,000
10M       │ 8ms         │ 97%       │ 3,000
50M       │ 15ms        │ 95%       │ 1,500
```

## Interview Questions

1. **Q: 50M vectors, 1000 QPS, <20ms latency. Architecture?**
   A: Weaviate on EKS (or Qdrant): 3 shards × 2 replicas = 6 nodes. Each node: r6i.4xlarge (128GB RAM). HNSW params: M=32, ef_construction=512, ef_search=256. Sharding: hash-based (even distribution). Load balancer routes reads to any replica. Expected: ~12ms p50, ~20ms p99 at 1000 QPS with 95%+ recall.

2. **Q: HNSW index build mein 6 hours lag rahe hain 10M vectors pe. Kaise speed up?**
   A: (1) Reduce ef_construction (512→256, faster build, slightly lower recall), (2) Parallel index building (if DB supports), (3) Build index in background (non-blocking), (4) Use IVFFlat for initial load, then rebuild as HNSW offline, (5) Batch inserts (bulk API, not one-by-one), (6) More RAM (avoids disk I/O during build).

3. **Q: Multi-tenant RAG system — each customer's data isolated. Design?**
   A: Option 1: Separate collection/index per tenant (Weaviate class per customer). Pros: Complete isolation, easy deletion. Cons: Many small indexes. Option 2: Single collection with tenant_id filter + metadata filtering. Pros: Simpler ops. Cons: Less isolation, filter overhead. Recommendation: <100 tenants → separate collections. >100 tenants → shared collection with partition key.
