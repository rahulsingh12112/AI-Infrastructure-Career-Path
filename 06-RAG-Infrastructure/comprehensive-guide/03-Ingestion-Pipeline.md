# 03 — Data Ingestion Pipeline

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              RAG Ingestion Pipeline                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  S3 Bucket (source)                                              │
│  ├── PDFs, DOCX, HTML, TXT, CSV                                │
│  └── New file uploaded → S3 Event                               │
│       │                                                          │
│       ▼                                                          │
│  EventBridge / S3 Notification                                   │
│       │                                                          │
│       ▼                                                          │
│  Step Functions (orchestration)                                  │
│  ├── Step 1: Extract text (Textract for PDFs, parsers)          │
│  ├── Step 2: Chunk (fixed-size / semantic)                      │
│  ├── Step 3: Embed (Bedrock Titan / self-hosted)               │
│  └── Step 4: Upsert to Vector DB                               │
│       │                                                          │
│       ▼                                                          │
│  Vector DB (OpenSearch / pgvector / Weaviate)                   │
│                                                                  │
│  Alternative: Lambda (simple) or ECS/EKS (heavy processing)    │
└─────────────────────────────────────────────────────────────────┘
```

## Chunking Strategies

```
Strategy         │ How it works                        │ Best For
─────────────────┼─────────────────────────────────────┼──────────────────
Fixed-size       │ Split every N tokens (with overlap) │ General purpose
Semantic         │ Split at paragraph/section breaks   │ Structured docs
Recursive        │ Try large splits, recurse smaller   │ Mixed content
Sentence-based   │ Split at sentence boundaries        │ QA, short answers
Document-based   │ Each document = 1 chunk             │ Short documents

Recommended:
├── Chunk size: 256-512 tokens (sweet spot for RAG)
├── Overlap: 10-20% (prevents losing context at boundaries)
├── Include metadata: source file, page number, section heading
└── Test: Try different sizes, evaluate retrieval quality
```

### Chunking Code
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)

chunks = splitter.split_text(document_text)
# Each chunk: ~512 chars with 50 char overlap
```

## Complete Pipeline (Step Functions + Lambda)

```python
# Lambda: Chunk + Embed + Store
import boto3
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

bedrock = boto3.client("bedrock-runtime")
opensearch_client = get_opensearch_client()

def handler(event, context):
    bucket = event["bucket"]
    key = event["key"]
    
    # 1. Extract text
    text = extract_text_from_s3(bucket, key)
    
    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    # 3. Embed (batch)
    embeddings = []
    for batch in batched(chunks, 25):  # Bedrock limit: 25 per call
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({"inputText": batch, "dimensions": 1536}),
        )
        embeddings.extend(json.loads(response["body"].read())["embeddings"])
    
    # 4. Store in OpenSearch
    actions = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        actions.append({"index": {"_index": "rag-vectors"}})
        actions.append({
            "text": chunk,
            "embedding": embedding,
            "metadata": {"source": key, "chunk_index": i},
        })
    
    opensearch_client.bulk(body=actions)
    return {"chunks_processed": len(chunks)}
```

## Batch Ingestion (Large Scale)

```python
# For millions of documents: Use ECS/EKS batch job

# Architecture:
# S3 (100K files) → SQS Queue → ECS Workers (10 containers) → Vector DB

# Worker processes files from SQS in parallel
# Throughput: 10 workers × 100 chunks/sec = 1000 chunks/sec
# 1M documents × 10 chunks each = 10M chunks / 1000 per sec = ~3 hours

# ECS Task Definition
{
    "containerDefinitions": [{
        "name": "ingestion-worker",
        "image": "my-ingestion:latest",
        "cpu": 2048,
        "memory": 4096,
        "environment": [
            {"name": "SQS_QUEUE_URL", "value": "https://sqs.us-east-1.amazonaws.com/..."},
            {"name": "VECTOR_DB_HOST", "value": "opensearch.internal"},
            {"name": "EMBEDDING_MODEL", "value": "amazon.titan-embed-text-v2:0"},
        ]
    }]
}
```

## Incremental Updates

```
Problem: Documents change. How to keep Vector DB in sync?

Strategy 1: Full re-index (simple, expensive)
├── Delete all vectors for document
├── Re-chunk, re-embed, re-insert
└── Good for: Small datasets, infrequent changes

Strategy 2: Change detection (efficient)
├── Track document hash (MD5/SHA)
├── On upload: Compare hash with stored hash
├── If changed: Delete old chunks, insert new ones
├── If unchanged: Skip
└── Good for: Large datasets, frequent changes

Strategy 3: Versioned chunks
├── Never delete, add version field
├── Query filters: version = latest
├── Old versions kept for audit/rollback
└── Good for: Compliance-heavy environments

# Change detection implementation:
import hashlib

def process_document(bucket, key):
    content = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    new_hash = hashlib.md5(content).hexdigest()
    
    # Check stored hash
    stored_hash = metadata_db.get_hash(key)
    if stored_hash == new_hash:
        return "SKIP"  # No change
    
    # Document changed: Re-process
    delete_old_chunks(key)
    chunks = chunk_and_embed(content)
    insert_chunks(chunks)
    metadata_db.update_hash(key, new_hash)
    return "UPDATED"
```

## Interview Questions

1. **Q: 1M PDFs ingest karne hain Vector DB mein. Architecture?**
   A: (1) S3 stores PDFs, (2) SQS queue with file paths, (3) 20× ECS workers pulling from SQS, (4) Each worker: Textract → chunk → batch embed (Bedrock Titan) → bulk insert OpenSearch, (5) Throughput: ~2000 chunks/sec, (6) 1M PDFs × 10 chunks = 10M chunks / 2000/sec = ~80 minutes. (7) Cost: ECS workers + Bedrock embedding API ($0.02/1M tokens) + OpenSearch storage.

2. **Q: Chunk size kya hona chahiye?**
   A: 256-512 tokens for most RAG use cases. Too small (<128): loses context, too many chunks retrieved. Too large (>1024): dilutes relevant info with noise, embedding less precise. Test with your data: (1) Try 256, 512, 1024, (2) Run retrieval eval (is correct chunk in top-5?), (3) Usually 512 with 20% overlap wins.

3. **Q: Documents update ho rahe hain daily. Stale data kaise handle karoge?**
   A: (1) S3 event → triggers re-ingestion pipeline, (2) Hash-based change detection (skip unchanged docs), (3) Delete old chunks by source_document ID, insert new, (4) Use metadata timestamp for freshness-based retrieval (prefer recent), (5) Schedule nightly full re-sync as safety net.
