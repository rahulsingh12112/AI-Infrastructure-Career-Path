# 03 — HuggingFace Text Generation Inference (TGI)

## What is TGI?

HuggingFace's production-ready LLM serving solution. Easiest way to serve HuggingFace models.

```
TGI = Production LLM server by HuggingFace:
├── Optimized for text generation (decoder models)
├── Continuous batching
├── Flash Attention & PagedAttention
├── Quantization (AWQ, GPTQ, BitsAndBytes)
├── Tensor parallelism (multi-GPU)
├── Token streaming (Server-Sent Events)
├── HuggingFace Hub integration (1-command deploy)
└── Used by: HuggingFace Inference API, many startups
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TGI Architecture                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  HTTP API:                                                   │
│  ├── POST /generate (single completion)                     │
│  ├── POST /generate_stream (streaming SSE)                  │
│  ├── POST /v1/chat/completions (OpenAI-compatible)          │
│  └── GET /info, /health, /metrics                           │
│                                                              │
│  Router (Rust — high performance):                           │
│  ├── Request validation                                      │
│  ├── Token counting                                          │
│  ├── Queue management                                        │
│  ├── Batching logic                                         │
│  └── Response streaming                                      │
│                                                              │
│  Model Server (Python + custom CUDA):                       │
│  ├── Flash Attention 2                                       │
│  ├── PagedAttention                                          │
│  ├── Continuous batching                                     │
│  ├── Quantized inference (AWQ/GPTQ/EETQ)                   │
│  └── Custom CUDA kernels (fused operations)                 │
│                                                              │
│  Model Loading:                                              │
│  ├── Direct from HuggingFace Hub                            │
│  ├── Safetensors format (fast, safe)                        │
│  ├── Sharded loading (for multi-GPU)                        │
│  └── Weight format: BF16, FP16, GPTQ, AWQ                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Running TGI

### Basic (Single GPU)
```bash
# Serve LLaMA-7B on single GPU
docker run --gpus all --shm-size 1g \
  -p 8080:80 \
  -v /data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-chat-hf \
  --max-input-length 2048 \
  --max-total-tokens 4096 \
  --max-batch-prefill-tokens 4096

# Test
curl localhost:8080/generate \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is deep learning?","parameters":{"max_new_tokens":100}}'

# Streaming
curl localhost:8080/generate_stream \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"Explain AI","parameters":{"max_new_tokens":200}}'
```

### Multi-GPU (Tensor Parallel)
```bash
docker run --gpus all --shm-size 1g \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-70b-chat-hf \
  --num-shard 4 \              # 4 GPU tensor parallel
  --max-input-length 2048 \
  --max-total-tokens 4096 \
  --quantize awq               # Quantized model
```

### Quantized Models
```bash
# AWQ quantized (4-bit, best quality/speed)
docker run --gpus all --shm-size 1g \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id TheBloke/Llama-2-70B-Chat-AWQ \
  --quantize awq \
  --max-input-length 2048 \
  --max-total-tokens 4096

# GPTQ quantized
docker run --gpus all --shm-size 1g \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id TheBloke/Llama-2-7B-Chat-GPTQ \
  --quantize gptq \
  --max-input-length 2048 \
  --max-total-tokens 4096
```

## All Configuration Options

```bash
docker run ghcr.io/huggingface/text-generation-inference:latest \
  --model-id <model>            # HuggingFace model ID or local path
  --revision <branch>           # Model revision/branch
  --num-shard <N>               # Tensor parallel GPUs
  --quantize <method>           # awq, gptq, bitsandbytes, eetq, fp8
  --dtype <type>                # float16, bfloat16
  --max-input-length <N>        # Max input tokens (default: 1024)
  --max-total-tokens <N>        # Max input + output tokens
  --max-batch-prefill-tokens <N># Max tokens in prefill batch
  --max-batch-total-tokens <N>  # Max tokens across all requests in batch
  --max-concurrent-requests <N> # Max concurrent requests (default: 128)
  --max-waiting-tokens <N>      # Max tokens waiting before new batch
  --waiting-served-ratio <F>    # Ratio for scheduling (prefill vs decode)
  --port <N>                    # Server port (default: 80)
  --trust-remote-code           # Allow custom model code
  --disable-custom-kernels      # Use generic kernels (debug)
  --rope-scaling <type>         # dynamic, linear (for long context)
  --rope-factor <F>             # RoPE scaling factor
```

## TGI API Endpoints

### Generate (Non-streaming)
```python
import requests

response = requests.post(
    "http://localhost:8080/generate",
    json={
        "inputs": "What is machine learning?",
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "seed": 42,
            "stop": ["\n\n"],           # Stop sequences
            "return_full_text": False,   # Don't include input in output
        },
    },
)
result = response.json()
print(result["generated_text"])
# Also returns: details.generated_tokens, details.finish_reason, etc.
```

### Streaming (Server-Sent Events)
```python
import requests
import json

response = requests.post(
    "http://localhost:8080/generate_stream",
    json={
        "inputs": "Explain quantum computing",
        "parameters": {"max_new_tokens": 500, "temperature": 0.7},
    },
    stream=True,
)

for line in response.iter_lines():
    if line:
        data = line.decode("utf-8")
        if data.startswith("data:"):
            token_data = json.loads(data[5:])
            print(token_data["token"]["text"], end="", flush=True)
```

### OpenAI-Compatible Endpoint
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is AI?"},
    ],
    max_tokens=200,
    temperature=0.7,
    stream=True,
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## TGI Performance Tuning

```
Parameter Tuning Guide:

1. --max-batch-prefill-tokens (MOST IMPORTANT)
   ├── Controls how many tokens are processed in prefill phase
   ├── Higher = more throughput (bigger batches)
   ├── Too high = OOM or high latency for individual requests
   ├── Default: 4096
   └── Recommendation: Start at 4096, increase until latency SLA met

2. --max-concurrent-requests
   ├── How many requests can be in-flight simultaneously
   ├── Higher = more queuing capacity
   ├── Limited by GPU memory (KV-cache)
   └── Default: 128

3. --waiting-served-ratio
   ├── Balance between serving new prefills vs existing decodes
   ├── Higher = prioritize new requests (lower TTFT)
   ├── Lower = prioritize completing existing requests (lower TPS)
   └── Default: 1.2

4. --max-waiting-tokens
   ├── How many decode tokens before scheduling next prefill batch
   ├── Lower = more responsive to new requests
   ├── Higher = less interruption for ongoing generation
   └── Default: 20
```

## TGI vs vLLM Comparison

```
Feature              │ TGI                    │ vLLM
─────────────────────┼────────────────────────┼────────────────────
Ease of use          │ ⭐⭐⭐⭐⭐ (1 docker cmd)  │ ⭐⭐⭐⭐ (slightly more config)
HuggingFace models   │ Native support         │ Good support
Throughput           │ ⭐⭐⭐⭐                  │ ⭐⭐⭐⭐⭐ (slightly better)
Streaming            │ Excellent (SSE native) │ Good
Quantization         │ AWQ, GPTQ, BnB, EETQ  │ AWQ, GPTQ, FP8
Multi-modal          │ Limited                │ Better support
Prefix caching       │ Limited                │ Excellent
Custom models        │ HuggingFace format only│ More flexible
Production features  │ Good                   │ Good
Router performance   │ Rust (very fast)       │ Python (slightly slower)
Community            │ HuggingFace ecosystem  │ Growing fast
Best for             │ Quick HF model deploy  │ Maximum throughput
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tgi-llama
spec:
  replicas: 2
  selector:
    matchLabels:
      app: tgi-llama
  template:
    metadata:
      labels:
        app: tgi-llama
    spec:
      containers:
      - name: tgi
        image: ghcr.io/huggingface/text-generation-inference:latest
        args:
        - "--model-id"
        - "meta-llama/Llama-2-7b-chat-hf"
        - "--max-input-length"
        - "2048"
        - "--max-total-tokens"
        - "4096"
        - "--max-batch-prefill-tokens"
        - "4096"
        ports:
        - containerPort: 80
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: token
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
        - name: model-cache
          mountPath: /data
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 1Gi
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: tgi-service
spec:
  selector:
    app: tgi-llama
  ports:
  - port: 8080
    targetPort: 80
```

## Monitoring TGI

```bash
# Health check
curl localhost:8080/health
# Returns: 200 OK when ready

# Model info
curl localhost:8080/info
# Returns: model_id, dtype, max_input_length, etc.

# Prometheus metrics
curl localhost:8080/metrics
# Key metrics:
# tgi_request_count                — total requests
# tgi_request_duration_seconds     — latency histogram
# tgi_request_generated_tokens     — tokens generated
# tgi_batch_current_size           — current batch size
# tgi_queue_size                   — requests waiting
# tgi_request_input_length         — input token histogram
```

## Interview Questions

1. **Q: TGI mein "prefill" aur "decode" phase kya hai?**
   A: Prefill = process ALL input tokens in one forward pass (compute-bound, parallel). Decode = generate output tokens one-by-one (memory-bound, sequential). TGI schedules them separately because they have different resource profiles. Prefill needs compute, decode needs memory bandwidth. --waiting-served-ratio controls the balance.

2. **Q: TGI 70B model serve karna hai 2× A100 pe. Configuration?**
   A: `--model-id meta-llama/Llama-2-70b-chat-hf --num-shard 2 --dtype bfloat16 --max-input-length 2048 --max-total-tokens 4096 --max-batch-prefill-tokens 4096 --max-concurrent-requests 64`. 70B in BF16 = 140GB across 2×80GB A100. Leaves ~20GB for KV-cache. Support ~50-64 concurrent requests with avg 512 tokens.

3. **Q: TGI queue size badhta ja raha hai, latency spike ho rahi hai. Debug?**
   A: (1) Check --max-concurrent-requests (too low?), (2) Check --max-batch-prefill-tokens (too low = small batches = low throughput), (3) GPU memory full → can't fit more requests → reduce max-total-tokens, (4) Scale horizontally (more replicas behind load balancer), (5) Check if specific long requests are blocking (set max-input-length limit).

4. **Q: TGI vs SageMaker endpoint for LLM serving?**
   A: TGI: Self-managed, maximum control, cheaper at scale, requires infra expertise. SageMaker: Managed (auto-scaling, monitoring, A/B testing), higher cost, less configuration needed. For startups/small teams: SageMaker (less ops burden). For scale/cost-sensitive: TGI on EKS (full control).
