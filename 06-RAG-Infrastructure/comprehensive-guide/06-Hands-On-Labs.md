# 06 — Hands-On Labs (RAG Infrastructure)

## Lab 1: Deploy Weaviate on EKS

```bash
# 1. Add Weaviate Helm repo
helm repo add weaviate https://weaviate.github.io/weaviate-helm
helm repo update

# 2. Install with persistent storage
helm install weaviate weaviate/weaviate \
  --set replicas=3 \
  --set storage.size=50Gi \
  --set storage.storageClassName=gp3 \
  --set env.PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  --set env.DEFAULT_VECTORIZER_MODULE=none \
  --set env.CLUSTER_DATA_BIND_PORT=7001

# 3. Verify
kubectl get pods -l app=weaviate
kubectl port-forward svc/weaviate 8080:80

# 4. Test
curl http://localhost:8080/v1/meta
```

```python
# Create class and insert data
import weaviate

client = weaviate.Client("http://localhost:8080")

client.schema.create_class({
    "class": "Document",
    "vectorizer": "none",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]},
    ],
})

# Insert with vector
client.data_object.create(
    class_name="Document",
    data_object={"content": "AI is transforming industries", "source": "doc1.pdf"},
    vector=[0.1] * 1536,  # Replace with real embedding
)

# Search
result = client.query.get("Document", ["content"])\
    .with_near_vector({"vector": [0.1] * 1536})\
    .with_limit(5).do()
print(result)
```

### Verification
- [ ] 3 Weaviate pods running
- [ ] Can create class and insert vectors
- [ ] Search returns relevant results
- [ ] Data persists after pod restart

---

## Lab 2: Set up pgvector on Aurora

```bash
# 1. Create Aurora PostgreSQL cluster (via AWS Console or CLI)
aws rds create-db-cluster \
  --db-cluster-identifier rag-aurora \
  --engine aurora-postgresql \
  --engine-version 15.4 \
  --master-username admin \
  --master-user-password MyPassword123 \
  --db-subnet-group-name my-subnet-group

# 2. Connect and enable pgvector
psql -h rag-aurora.cluster-xxx.us-east-1.rds.amazonaws.com -U admin -d postgres
```

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create HNSW index
CREATE INDEX docs_embedding_idx ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 256);

-- Insert test data
INSERT INTO documents (content, metadata, embedding)
VALUES ('AI is the future', '{"source": "test.pdf"}', '[0.1, 0.2, ...]'::vector);

-- Search
SELECT content, 1 - (embedding <=> '[0.15, 0.25, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.15, 0.25, ...]'::vector
LIMIT 5;
```

### Verification
- [ ] Aurora cluster running with pgvector enabled
- [ ] Can insert vectors and search
- [ ] HNSW index created successfully
- [ ] Search latency < 10ms for 100K vectors

---

## Lab 3: Build Ingestion Pipeline (S3 → Chunk → Embed → Store)

```python
# ingestion_pipeline.py — Complete pipeline
import boto3
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from opensearchpy import OpenSearch

bedrock = boto3.client("bedrock-runtime")
s3 = boto3.client("s3")

def ingest_document(bucket, key):
    # 1. Download from S3
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    
    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks from {key}")
    
    # 3. Embed each chunk
    embeddings = []
    for chunk in chunks:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": chunk, "dimensions": 1536, "normalize": True}),
        )
        emb = json.loads(response["body"].read())["embedding"]
        embeddings.append(emb)
    
    # 4. Store in vector DB
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        store_vector(chunk, embedding, {"source": key, "chunk_idx": i})
    
    print(f"Ingested {len(chunks)} vectors for {key}")

# Run
ingest_document("my-docs-bucket", "documents/policy.txt")
```

### Verification
- [ ] Document downloaded from S3
- [ ] Chunked into appropriate sizes
- [ ] Embeddings generated via Bedrock
- [ ] Vectors stored in DB and searchable

---

## Lab 4: Set up Bedrock Knowledge Base

```python
import boto3
import time

bedrock_agent = boto3.client("bedrock-agent")

# Create KB (with OpenSearch Serverless backend)
# Note: Need to create OpenSearch collection first

kb = bedrock_agent.create_knowledge_base(
    name="lab-knowledge-base",
    roleArn="arn:aws:iam::123456:role/BedrockKBRole",
    knowledgeBaseConfiguration={
        "type": "VECTOR",
        "vectorKnowledgeBaseConfiguration": {
            "embeddingModelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
        },
    },
    storageConfiguration={
        "type": "OPENSEARCH_SERVERLESS",
        "opensearchServerlessConfiguration": {
            "collectionArn": "arn:aws:aoss:...",
            "vectorIndexName": "lab-index",
            "fieldMapping": {
                "vectorField": "embedding",
                "textField": "text",
                "metadataField": "metadata",
            },
        },
    },
)

kb_id = kb["knowledgeBase"]["knowledgeBaseId"]

# Add data source
ds = bedrock_agent.create_data_source(
    knowledgeBaseId=kb_id,
    name="s3-source",
    dataSourceConfiguration={
        "type": "S3",
        "s3Configuration": {"bucketArn": "arn:aws:s3:::my-docs-bucket"},
    },
)

# Sync
bedrock_agent.start_ingestion_job(
    knowledgeBaseId=kb_id,
    dataSourceId=ds["dataSource"]["dataSourceId"],
)

# Query
bedrock_runtime = boto3.client("bedrock-agent-runtime")
response = bedrock_runtime.retrieve(
    knowledgeBaseId=kb_id,
    retrievalQuery={"text": "What is the refund policy?"},
    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}},
)

for result in response["retrievalResults"]:
    print(f"Score: {result['score']:.3f} | {result['content']['text'][:100]}")
```

### Verification
- [ ] Knowledge Base created
- [ ] Data source synced
- [ ] Queries return relevant results
- [ ] Citations point to correct source documents

---

## Lab 5: Benchmark Vector Search at Scale

```python
import time
import numpy as np

def benchmark_vector_search(client, num_vectors, num_queries=100):
    """Benchmark search latency at different scales"""
    
    # Insert vectors
    print(f"Inserting {num_vectors} vectors...")
    vectors = np.random.randn(num_vectors, 1536).astype(np.float32)
    
    start = time.time()
    for i in range(0, num_vectors, 1000):
        batch = vectors[i:i+1000]
        bulk_insert(client, batch)  # Your bulk insert function
    insert_time = time.time() - start
    print(f"Insert time: {insert_time:.1f}s ({num_vectors/insert_time:.0f} vec/s)")
    
    # Search benchmark
    query_vectors = np.random.randn(num_queries, 1536).astype(np.float32)
    latencies = []
    
    for qv in query_vectors:
        start = time.time()
        results = search(client, qv, k=10)  # Your search function
        latencies.append((time.time() - start) * 1000)
    
    print(f"\nSearch Results ({num_vectors} vectors, {num_queries} queries):")
    print(f"  P50: {np.percentile(latencies, 50):.1f}ms")
    print(f"  P90: {np.percentile(latencies, 90):.1f}ms")
    print(f"  P99: {np.percentile(latencies, 99):.1f}ms")
    print(f"  QPS: {num_queries / sum(latencies) * 1000:.0f}")

# Run at different scales
for n in [1000, 10000, 100000, 1000000]:
    benchmark_vector_search(client, n)
```

### Expected Results
```
Vectors  │ P50 (ms) │ P99 (ms) │ QPS
─────────┼──────────┼──────────┼──────
1K       │ 1        │ 3        │ 10,000
10K      │ 2        │ 5        │ 5,000
100K     │ 4        │ 12       │ 2,000
1M       │ 8        │ 25       │ 800
```

### Verification
- [ ] Benchmarks run at 4 scale levels
- [ ] Latency documented for each
- [ ] Identified scale where performance degrades
- [ ] Can recommend hardware sizing based on results

---

## Lab 6: Implement Semantic Caching

```python
import redis
import numpy as np
import json
import hashlib

r = redis.Redis(host="localhost", port=6379)

class RAGCache:
    def __init__(self, threshold=0.93):
        self.threshold = threshold
    
    def _exact_key(self, query):
        return f"exact:{hashlib.md5(query.lower().strip().encode()).hexdigest()}"
    
    def get(self, query, query_embedding):
        # 1. Try exact match
        exact = r.get(self._exact_key(query))
        if exact:
            return json.loads(exact), "exact_hit"
        
        # 2. Try semantic match (simplified — use Redis Vector in production)
        # Store embeddings and check similarity
        cached_keys = r.keys("semantic:*")
        for key in cached_keys[:100]:  # Check last 100
            cached = json.loads(r.get(key))
            sim = np.dot(query_embedding, cached["embedding"]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(cached["embedding"])
            )
            if sim >= self.threshold:
                return cached["answer"], "semantic_hit"
        
        return None, "miss"
    
    def store(self, query, query_embedding, answer):
        # Store exact
        r.setex(self._exact_key(query), 86400, json.dumps(answer))
        # Store semantic
        r.setex(
            f"semantic:{hashlib.md5(query.encode()).hexdigest()}",
            86400,
            json.dumps({"embedding": query_embedding.tolist(), "answer": answer}),
        )

# Usage
cache = RAGCache(threshold=0.93)
query = "What is the return policy?"
embedding = get_embedding(query)

result, hit_type = cache.get(query, embedding)
if result:
    print(f"Cache {hit_type}: {result}")
else:
    answer = run_full_rag(query)
    cache.store(query, embedding, answer)
    print(f"Cache miss, stored: {answer}")
```

### Verification
- [ ] Exact match cache works (same query → hit)
- [ ] Semantic cache works (similar query → hit)
- [ ] Cache miss triggers full RAG pipeline
- [ ] Results stored for future queries
- [ ] Measure hit rate over 100 queries

---

## Summary Checklist
- [ ] Lab 1: Weaviate on EKS (3 replicas, persistent)
- [ ] Lab 2: pgvector on Aurora (HNSW index, search working)
- [ ] Lab 3: Ingestion pipeline (S3 → chunk → embed → store)
- [ ] Lab 4: Bedrock Knowledge Base (managed RAG)
- [ ] Lab 5: Benchmarks at 1K/10K/100K/1M vectors
- [ ] Lab 6: Semantic + exact caching implemented
