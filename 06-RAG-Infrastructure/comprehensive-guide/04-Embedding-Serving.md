# 04 — Embedding Model Serving

## Embedding Options

```
Option                      │ Dimensions │ Speed    │ Cost          │ Quality
────────────────────────────┼────────────┼──────────┼───────────────┼─────────
Bedrock Titan Embed v2      │ 256-1536   │ Fast     │ $0.02/1M tok  │ Good
OpenAI text-embedding-3-large│ 256-3072  │ Fast     │ $0.13/1M tok  │ Best
Self-hosted (BGE-large)     │ 1024       │ Variable │ GPU cost only │ Great
Self-hosted (E5-mistral-7B) │ 4096       │ Slow     │ GPU cost only │ Excellent
Cohere embed-v3             │ 1024       │ Fast     │ $0.10/1M tok  │ Great
```

## Bedrock Titan Embeddings (Managed)

```python
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def get_embedding(text, dimensions=1536):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({
            "inputText": text,
            "dimensions": dimensions,    # 256, 512, 1024, or 1536
            "normalize": True,
        }),
    )
    return json.loads(response["body"].read())["embedding"]

# Batch embedding (up to 25 texts)
def get_embeddings_batch(texts):
    embeddings = []
    for text in texts:
        embeddings.append(get_embedding(text))
    return embeddings
    # Note: Bedrock doesn't support true batch API yet
    # Use concurrent requests for parallelism
```

## Self-hosted on SageMaker

```python
from sagemaker.huggingface import HuggingFaceModel

# Deploy embedding model on SageMaker
hub_config = {
    "HF_MODEL_ID": "BAAI/bge-large-en-v1.5",
    "HF_TASK": "feature-extraction",
}

model = HuggingFaceModel(
    env=hub_config,
    role=role,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
)

predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.g5.xlarge",     # GPU for fast embedding
    endpoint_name="embedding-endpoint",
)

# Invoke
response = predictor.predict({
    "inputs": ["What is AI?", "Machine learning basics"],
})
# Returns: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
```

## Self-hosted on EKS (TEI — Text Embeddings Inference)

```yaml
# HuggingFace TEI — optimized embedding server
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embedding-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: tei
        image: ghcr.io/huggingface/text-embeddings-inference:latest
        args:
        - "--model-id"
        - "BAAI/bge-large-en-v1.5"
        - "--max-batch-tokens"
        - "16384"
        - "--max-concurrent-requests"
        - "512"
        ports:
        - containerPort: 80
        resources:
          limits:
            nvidia.com/gpu: 1
```

```python
# Client usage
import requests

response = requests.post(
    "http://embedding-service/embed",
    json={"inputs": ["What is AI?", "Explain RAG"]},
)
embeddings = response.json()  # List of vectors
```

## Batch Embedding for Bulk Ingestion

```python
import asyncio
import aiohttp

async def embed_batch_async(texts, batch_size=64, max_concurrent=10):
    """Embed large dataset efficiently with concurrent batches"""
    semaphore = asyncio.Semaphore(max_concurrent)
    all_embeddings = []
    
    async def process_batch(batch):
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://embedding-service/embed",
                    json={"inputs": batch},
                ) as response:
                    return await response.json()
    
    # Split into batches
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    # Process concurrently
    tasks = [process_batch(batch) for batch in batches]
    results = await asyncio.gather(*tasks)
    
    for result in results:
        all_embeddings.extend(result)
    
    return all_embeddings

# Embed 1M chunks efficiently
# 10 concurrent × 64 per batch = 640 embeddings in flight
# TEI on GPU: ~5000 embeddings/sec
# 1M chunks / 5000/sec = ~200 seconds = 3.3 minutes!
```

## Cost Comparison

```
Scenario: Embed 10M chunks (500 tokens avg each)
Total tokens: 5B tokens

Bedrock Titan: 5000M tokens × $0.02/1M = $100
OpenAI:        5000M tokens × $0.13/1M = $650
Self-hosted:   g5.xlarge × 1 hr = $1.00 (processes 5000 emb/sec = 3.3 hrs = $3.30)

For bulk ingestion: Self-hosted is 30x cheaper than Bedrock!
For real-time (low volume): Bedrock is simpler (no infra)

Decision:
├── < 1M embeddings/month → Bedrock (simpler, $2/month)
├── 1M-100M/month → Either (cost similar)
├── > 100M/month → Self-hosted (10-30x cheaper)
└── Burst ingestion (10M at once) → Self-hosted (parallel processing)
```

## Interview Questions

1. **Q: Real-time RAG query mein embedding latency add ho rahi hai. Kaise optimize?**
   A: (1) Use smaller embedding model for queries (BGE-small = 3x faster, slightly less quality), (2) GPU-accelerated TEI server (5000 emb/sec vs 100 on CPU), (3) Connection pooling to embedding service, (4) Cache frequent query embeddings (Redis), (5) Reduce dimensions (1536→512 with Titan v2, minimal quality loss).

2. **Q: Embedding model upgrade karna hai (v1 → v2). Vectors re-embed karne padenge?**
   A: Yes! Different embedding models produce incompatible vectors. Strategy: (1) Create new index with v2 embeddings, (2) Backfill in background (batch process all docs), (3) Once complete: switch queries to new index, (4) Delete old index. Never mix embeddings from different models in same index!
