# 01 — Vector DB Deployment Options on AWS

## What is a Vector Database?

Stores embeddings (numerical representations of text/images) and enables similarity search.

```
Traditional DB: SELECT * FROM docs WHERE title = "AI"  (exact match)
Vector DB:      Find 10 documents MOST SIMILAR to this query embedding  (semantic search)

How RAG uses it:
1. Documents → chunked → embedded → stored in Vector DB
2. User query → embedded → search Vector DB → get relevant chunks
3. Relevant chunks + query → LLM → grounded answer
```

## Options Comparison

```
Option                │ Type        │ Scale    │ Cost       │ Best For
──────────────────────┼─────────────┼──────────┼────────────┼──────────────────────
Bedrock Knowledge Base│ Managed     │ Auto     │ Pay/query  │ Quick start, no ops
OpenSearch Serverless │ Managed     │ Auto     │ Medium     │ AWS-native, hybrid search
pgvector on Aurora    │ Self-managed│ Manual   │ Low        │ Already using PostgreSQL
Pinecone             │ SaaS        │ Auto     │ High       │ Simplest, fastest start
Weaviate on EKS      │ Self-hosted │ Manual   │ Low        │ Full control, hybrid
Qdrant on EKS        │ Self-hosted │ Manual   │ Low        │ Performance focused
Milvus on EKS        │ Self-hosted │ Manual   │ Low        │ Massive scale (billions)
```

## 1. Bedrock Knowledge Bases (Fully Managed)

```
Architecture:
┌────────────────────────────────────────────────────┐
│  Bedrock Knowledge Base (fully managed)             │
│                                                     │
│  Data Sources (S3):                                 │
│  └── PDFs, docs, web → Auto chunking → Auto embed │
│                                                     │
│  Vector Store (choose one):                         │
│  ├── OpenSearch Serverless (default)                │
│  ├── Aurora PostgreSQL (pgvector)                   │
│  ├── Pinecone                                       │
│  └── Redis Enterprise                              │
│                                                     │
│  Query flow:                                        │
│  User → Bedrock → embed query → search vector DB   │
│  → retrieve chunks → augment prompt → LLM → answer │
└────────────────────────────────────────────────────┘
```

```python
import boto3

bedrock_agent = boto3.client("bedrock-agent", region_name="us-east-1")

# Create Knowledge Base
kb = bedrock_agent.create_knowledge_base(
    name="company-docs",
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
            "collectionArn": "arn:aws:aoss:us-east-1:123456:collection/my-kb",
            "vectorIndexName": "bedrock-kb-index",
            "fieldMapping": {
                "vectorField": "embedding",
                "textField": "text",
                "metadataField": "metadata",
            },
        },
    },
)

# Add S3 data source
bedrock_agent.create_data_source(
    knowledgeBaseId=kb["knowledgeBase"]["knowledgeBaseId"],
    name="s3-docs",
    dataSourceConfiguration={
        "type": "S3",
        "s3Configuration": {
            "bucketArn": "arn:aws:s3:::my-docs-bucket",
            "inclusionPrefixes": ["documents/"],
        },
    },
    vectorIngestionConfiguration={
        "chunkingConfiguration": {
            "chunkingStrategy": "FIXED_SIZE",
            "fixedSizeChunkingConfiguration": {
                "maxTokens": 512,
                "overlapPercentage": 20,
            },
        },
    },
)

# Sync (trigger ingestion)
bedrock_agent.start_ingestion_job(
    knowledgeBaseId=kb["knowledgeBase"]["knowledgeBaseId"],
    dataSourceId=data_source_id,
)

# Query
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime")
response = bedrock_agent_runtime.retrieve_and_generate(
    input={"text": "What is our refund policy?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": kb["knowledgeBase"]["knowledgeBaseId"],
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
        },
    },
)
print(response["output"]["text"])
```

## 2. OpenSearch Serverless (Vector Search)

```python
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

# Connect to OpenSearch Serverless
credentials = boto3.Session().get_credentials()
auth = AWS4Auth(credentials.access_key, credentials.secret_key,
                "us-east-1", "aoss", session_token=credentials.token)

client = OpenSearch(
    hosts=[{"host": "xxxxx.us-east-1.aoss.amazonaws.com", "port": 443}],
    http_auth=auth,
    use_ssl=True,
    connection_class=RequestsHttpConnection,
)

# Create vector index
index_body = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 512,
        }
    },
    "mappings": {
        "properties": {
            "embedding": {
                "type": "knn_vector",
                "dimension": 1536,        # Titan embedding dimension
                "method": {
                    "name": "hnsw",
                    "engine": "nmslib",
                    "space_type": "cosinesimil",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16,
                    },
                },
            },
            "text": {"type": "text"},
            "metadata": {"type": "object"},
        }
    },
}
client.indices.create(index="rag-vectors", body=index_body)

# Index a document
client.index(index="rag-vectors", body={
    "embedding": [0.1, 0.2, ...],  # 1536-dim vector
    "text": "Our refund policy allows returns within 30 days...",
    "metadata": {"source": "policy.pdf", "page": 5},
})

# Search
results = client.search(index="rag-vectors", body={
    "size": 5,
    "query": {
        "knn": {
            "embedding": {
                "vector": query_embedding,  # Query vector
                "k": 5,
            }
        }
    },
})
```

## 3. pgvector on Aurora PostgreSQL

```sql
-- Enable pgvector extension
CREATE EXTENSION vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(1536),    -- 1536 dimensions
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create HNSW index (faster search)
CREATE INDEX ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 256);

-- Insert document
INSERT INTO documents (content, metadata, embedding)
VALUES (
    'Our refund policy allows returns within 30 days...',
    '{"source": "policy.pdf", "page": 5}',
    '[0.1, 0.2, ...]'::vector
);

-- Search (cosine similarity, top 5)
SELECT content, metadata,
       1 - (embedding <=> '[0.15, 0.25, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.15, 0.25, ...]'::vector
LIMIT 5;
```

```python
# Python with psycopg2
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("host=aurora-cluster.xxx.us-east-1.rds.amazonaws.com dbname=ragdb")
register_vector(conn)

cur = conn.cursor()
cur.execute("""
    SELECT content, metadata,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
""", (query_embedding, query_embedding))

results = cur.fetchall()
```

## 4. Weaviate on EKS

```yaml
# Helm deployment
# helm install weaviate weaviate/weaviate -f values.yaml

# values.yaml
replicas: 3
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"

storage:
  size: 100Gi
  storageClassName: gp3

env:
  QUERY_DEFAULTS_LIMIT: 25
  PERSISTENCE_DATA_PATH: /var/lib/weaviate
  DEFAULT_VECTORIZER_MODULE: none    # We provide our own embeddings
  ENABLE_MODULES: ""
  CLUSTER_HOSTNAME: node1

# For GPU-accelerated search:
# resources.limits."nvidia.com/gpu": 1
```

```python
import weaviate

client = weaviate.Client("http://weaviate-service:8080")

# Create class (collection)
client.schema.create_class({
    "class": "Document",
    "vectorizer": "none",     # We handle embeddings
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]},
        {"name": "page", "dataType": ["int"]},
    ],
})

# Insert
client.data_object.create(
    class_name="Document",
    data_object={
        "content": "Our refund policy...",
        "source": "policy.pdf",
        "page": 5,
    },
    vector=[0.1, 0.2, ...],  # Pre-computed embedding
)

# Search
result = client.query.get("Document", ["content", "source"])\
    .with_near_vector({"vector": query_embedding})\
    .with_limit(5)\
    .with_additional(["distance"])\
    .do()
```

## Decision Matrix (When to Use What)

```
Scenario                                    │ Best Choice
────────────────────────────────────────────┼─────────────────────────
Prototype / POC (1 week deadline)           │ Bedrock Knowledge Base
Already using PostgreSQL + < 1M vectors     │ pgvector on Aurora
Need hybrid search (keyword + vector)       │ OpenSearch or Weaviate
Multi-tenant SaaS (isolated per customer)   │ Pinecone or Weaviate
Massive scale (100M+ vectors)              │ Milvus or Qdrant
Full control + K8s native                   │ Weaviate on EKS
Serverless (no capacity planning)           │ OpenSearch Serverless
Budget constrained + existing RDS           │ pgvector (free extension)
```

## Interview Questions

1. **Q: 10M documents ka RAG system banana hai. Vector DB kya choose karoge?**
   A: Depends on requirements. (1) If already on AWS + want managed: OpenSearch Serverless (scales automatically, hybrid search). (2) If need full control + multi-tenant: Weaviate on EKS (shard per tenant). (3) If using PostgreSQL already: pgvector on Aurora (but performance degrades >5M with HNSW). (4) If massive scale needed: Qdrant or Milvus (designed for billions). For 10M documents → OpenSearch Serverless is sweet spot (managed + scalable + hybrid).

2. **Q: pgvector vs dedicated vector DB (Weaviate/Qdrant)?**
   A: pgvector: Pros = no new infra, SQL queries, joins with relational data, free. Cons = slower at scale (>5M vectors), no built-in sharding for vectors, limited filtering performance. Dedicated: Pros = optimized for vector ops, better at scale, advanced features (hybrid search, multi-tenancy). Cons = new infra to manage. Rule: <1M vectors + already have Postgres → pgvector. >1M or performance critical → dedicated vector DB.

3. **Q: Bedrock Knowledge Base vs custom RAG pipeline?**
   A: Bedrock KB: Zero infra, auto chunking + embedding + indexing, managed updates, limited customization (chunking strategy, retrieval logic). Custom: Full control over chunking (semantic vs fixed), embedding model choice, retrieval strategy (hybrid, re-ranking), filtering, metadata. Choose Bedrock KB for: Standard use cases, quick deployment. Choose custom for: Complex retrieval needs, specific quality requirements, cost optimization at scale.
