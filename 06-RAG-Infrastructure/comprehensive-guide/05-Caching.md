# 05 — Caching for RAG

## Why Cache?

```
RAG request cost:
├── Embed query: 10-50ms + $0.00002
├── Vector search: 5-30ms
├── LLM generation: 500-3000ms + $0.001-0.01
└── Total: ~1-3 seconds, ~$0.01 per request

With caching:
├── Same/similar question asked before → return cached answer
├── Latency: 1-5ms (from cache)
├── Cost: $0 (no LLM call)
├── Cache hit rate: 20-40% typical for enterprise chatbots
└── Savings: 20-40% cost + latency reduction
```

## Two Types of Cache

```
┌─────────────────────────────────────────────────────────────┐
│                    Caching Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Query                                                  │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────┐                                           │
│  │ Exact Cache  │ (Redis/ElastiCache)                       │
│  │ hash(query)  │ → Hit? Return cached answer               │
│  └──────┬───────┘                                           │
│         │ Miss                                               │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │Semantic Cache│ (Vector similarity)                        │
│  │embed(query)  │ → Similar query found? Return its answer  │
│  └──────┬───────┘                                           │
│         │ Miss                                               │
│         ▼                                                    │
│  Full RAG pipeline (embed → search → LLM)                   │
│         │                                                    │
│         ▼                                                    │
│  Store in both caches for next time                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 1. Exact Match Cache (Redis/ElastiCache)

```python
import redis
import hashlib
import json

r = redis.Redis(host="elasticache-cluster.xxx.cache.amazonaws.com", port=6379)

def get_cache_key(query: str) -> str:
    """Normalize and hash query for exact match"""
    normalized = query.lower().strip()
    return f"rag:exact:{hashlib.md5(normalized.encode()).hexdigest()}"

def cached_rag(query: str):
    # Check exact cache
    cache_key = get_cache_key(query)
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)  # Cache HIT
    
    # Cache miss → full RAG
    answer = run_full_rag_pipeline(query)
    
    # Store with TTL (24 hours)
    r.setex(cache_key, 86400, json.dumps(answer))
    return answer
```

## 2. Semantic Cache (Similar Queries)

```python
import numpy as np

# Semantic cache using vector similarity
# If a SIMILAR question was asked before → reuse its answer

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.threshold = similarity_threshold
        self.cache_embeddings = []   # Stored query embeddings
        self.cache_answers = []      # Corresponding answers
        # In production: Use vector DB (Redis Vector, pgvector, etc.)
    
    def search(self, query_embedding):
        """Find similar cached query"""
        if not self.cache_embeddings:
            return None
        
        # Cosine similarity with all cached embeddings
        similarities = [
            np.dot(query_embedding, cached) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(cached))
            for cached in self.cache_embeddings
        ]
        
        max_sim = max(similarities)
        if max_sim >= self.threshold:
            idx = similarities.index(max_sim)
            return self.cache_answers[idx]  # Similar enough!
        return None  # No similar query found
    
    def store(self, query_embedding, answer):
        self.cache_embeddings.append(query_embedding)
        self.cache_answers.append(answer)

# Usage
cache = SemanticCache(similarity_threshold=0.92)

query = "What is our return policy?"
query_embedding = embed(query)

# Check semantic cache
cached_answer = cache.search(query_embedding)
if cached_answer:
    return cached_answer  # Semantic cache HIT!

# Full RAG
answer = run_full_rag_pipeline(query)
cache.store(query_embedding, answer)
return answer
```

### Production Semantic Cache (Redis + Vector)

```python
# Using Redis with vector search (RediSearch module)
import redis
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField

r = redis.Redis(host="redis-vector.xxx.cache.amazonaws.com", port=6379)

# Create index
r.ft("cache_idx").create_index([
    VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"}),
    TextField("answer"),
    TextField("query"),
])

# Store in cache
def cache_store(query, embedding, answer):
    r.hset(f"cache:{hash(query)}", mapping={
        "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        "answer": answer,
        "query": query,
    })

# Search cache
def cache_search(query_embedding, threshold=0.92):
    q = Query(f"*=>[KNN 1 @embedding $vec AS score]")\
        .return_fields("answer", "score")\
        .sort_by("score")\
        .dialect(2)
    
    params = {"vec": np.array(query_embedding, dtype=np.float32).tobytes()}
    results = r.ft("cache_idx").search(q, query_params=params)
    
    if results.docs and float(results.docs[0].score) >= threshold:
        return results.docs[0].answer
    return None
```

## Cache Invalidation

```
When to invalidate:
├── Source documents updated → cached answers may be stale
├── Time-based TTL (24h for dynamic data, 7d for static docs)
├── Model changed → regenerate all cached answers
└── Manual purge (admin force-refresh)

Strategies:
├── TTL-based: Simple, set expiry on cache entries
├── Event-driven: S3 doc update → purge related cache entries
├── Version tag: Include source_doc_version in cache key
└── Hybrid: TTL + event-driven (belt and suspenders)

# Event-driven invalidation
def on_document_updated(doc_id):
    """When source document changes, invalidate cached answers"""
    # Find all cache entries that used this document
    affected_keys = r.smembers(f"cache:doc_deps:{doc_id}")
    for key in affected_keys:
        r.delete(key)
```

## Cache Metrics

```
Key metrics to monitor:
├── Hit rate: % of requests served from cache (target: 30-50%)
├── Exact hit rate: Hash match hits
├── Semantic hit rate: Similar query hits
├── Miss latency: Time for full RAG on cache miss
├── Hit latency: Time to serve from cache (<5ms target)
├── Cache size: Memory usage (cost factor)
├── Stale rate: % of cached answers that are outdated
└── Eviction rate: Cache entries removed (memory pressure)

# Dashboard formula:
cost_savings = cache_hit_rate × llm_cost_per_request × total_requests
latency_improvement = cache_hit_rate × (full_rag_latency - cache_latency)
```

## Interview Questions

1. **Q: Semantic cache ka similarity threshold kya hona chahiye?**
   A: 0.92-0.95 for most use cases. Too low (0.85): Returns wrong cached answers for different questions. Too high (0.99): Almost never hits cache (only exact paraphrases). Tune with your data: (1) Collect query pairs that should/shouldn't match, (2) Find threshold that maximizes hits without wrong answers. Start conservative (0.95), lower gradually while monitoring quality.

2. **Q: Cache mein stale data aa raha hai. Users ko outdated answers mil rahe hain. Fix?**
   A: (1) Reduce TTL (24h → 4h for dynamic docs), (2) Event-driven invalidation (doc update → purge cache), (3) Include doc version in cache key (new version = cache miss), (4) Background re-validation (periodically re-run cached queries, update if different), (5) User feedback: "answer outdated" button → force invalidate.

3. **Q: Cache memory 50GB ho gaya. Kaise manage karoge?**
   A: (1) LRU eviction policy (remove least recently used), (2) TTL on all entries (auto-expire), (3) Tiered caching: Hot queries in Redis (fast, expensive), cold in S3 (slow, cheap), (4) Compress cached answers (gzip), (5) Only cache expensive queries (long answers, >$0.005 LLM cost), (6) Size cap: Redis maxmemory + eviction policy.
