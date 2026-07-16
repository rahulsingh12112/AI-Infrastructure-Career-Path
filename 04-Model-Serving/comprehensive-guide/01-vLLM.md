# 01 — vLLM: PagedAttention & Continuous Batching

## What is vLLM?

High-throughput LLM serving engine. 2-24x faster than naive HuggingFace inference.

```
Problem with naive LLM serving:
├── Each request allocates full KV-cache upfront (wastes GPU memory)
├── Requests processed one at a time (GPU idle between tokens)
├── Long sequences block short ones
├── GPU utilization: 10-30% (most memory wasted!)
└── Result: Expensive, slow, can't serve many users

vLLM solves this with:
├── PagedAttention: KV-cache managed like OS virtual memory (pages)
├── Continuous Batching: New requests added mid-generation
├── Memory efficiency: Near-zero waste
├── GPU utilization: 80-95%
└── Result: 2-24x more throughput, same hardware
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        vLLM Engine                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  API Layer (OpenAI-compatible):                              │
│  ├── /v1/completions                                        │
│  ├── /v1/chat/completions                                   │
│  └── /v1/models                                             │
│                                                              │
│  Scheduler:                                                  │
│  ├── Continuous batching (add/remove requests mid-batch)     │
│  ├── Priority scheduling                                    │
│  ├── Preemption (pause low-priority for high-priority)      │
│  └── Prefix caching (reuse KV-cache for shared prompts)     │
│                                                              │
│  Memory Manager (PagedAttention):                            │
│  ├── KV-cache split into fixed-size blocks (pages)          │
│  ├── Pages allocated on-demand (not upfront)                │
│  ├── Block table maps logical → physical blocks             │
│  ├── Copy-on-write for parallel sampling                    │
│  └── Memory fragmentation: ~0% (vs 60-80% without)         │
│                                                              │
│  Execution Engine:                                           │
│  ├── Custom CUDA kernels for paged attention                │
│  ├── Flash Attention integration                            │
│  ├── Tensor parallelism (multi-GPU)                        │
│  └── Quantization support (AWQ, GPTQ, FP8)                │
│                                                              │
│  Model Support:                                              │
│  ├── LLaMA, Mistral, Mixtral, Qwen, GPT-NeoX              │
│  ├── Falcon, MPT, OPT, BLOOM                               │
│  └── Vision-Language models (LLaVA, etc.)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## PagedAttention — Deep Dive

### The Problem: KV-Cache Memory Waste

```
Standard attention (without paging):
├── Request comes in: "Tell me about AI" (5 tokens)
├── Model generates up to max_tokens (say 2048)
├── GPU must allocate KV-cache for FULL 2048 tokens UPFRONT
├── But response might only be 100 tokens!
├── Wasted: 2048 - 100 = 1948 slots × 2 (K+V) × num_layers × hidden_size
└── With 7B model: ~1.5 GB wasted PER REQUEST!

10 concurrent requests = 15 GB wasted = 3-4 requests that could have fit

Standard memory layout:
┌─────────────────────────────────────────────┐
│ Request 1 KV-cache: [used  |   WASTED     ] │  ← Only 20% used!
│ Request 2 KV-cache: [used|     WASTED      ] │
│ Request 3 KV-cache: [used    |  WASTED     ] │
│ FREE SPACE: Can't fit Request 4!             │
└─────────────────────────────────────────────┘
```

### PagedAttention Solution

```
Inspired by OS virtual memory:
├── KV-cache divided into fixed-size BLOCKS (like memory pages)
├── Each block holds KV for N tokens (e.g., 16 tokens per block)
├── Blocks allocated ON-DEMAND as tokens generate
├── No pre-allocation = no waste!
├── Non-contiguous storage (blocks can be anywhere in GPU memory)
└── Block table maps logical position → physical GPU memory

PagedAttention memory layout:
┌─────────────────────────────────────────────┐
│ Physical GPU Memory (blocks):                │
│ [Req1-B0][Req2-B0][Req1-B1][Req3-B0]       │
│ [Req2-B1][FREE][Req3-B1][Req1-B2]          │
│ [FREE][FREE][Req2-B2][FREE]                 │
│                                              │
│ Block Tables:                                │
│ Req 1: logical [0,1,2] → physical [0,2,7]  │
│ Req 2: logical [0,1,2] → physical [1,4,10] │
│ Req 3: logical [0,1]   → physical [3,6]    │
│                                              │
│ Result: ~4% waste (only last block partial)  │
│ vs 60-80% waste without paging!             │
└─────────────────────────────────────────────┘
```

### Memory Savings

```
Model: LLaMA-13B, max_seq_len=2048

Without PagedAttention:
├── KV-cache per request: 2048 × 40 layers × 2(K+V) × 5120 dim × 2 bytes = 1.6 GB
├── A100 80GB: max ~40 concurrent requests (after model weights)
└── But average sequence = 256 tokens → 87.5% memory WASTED

With PagedAttention:
├── KV-cache per request: allocated only for actual tokens
├── Average: 256 tokens × same = 200 MB per request
├── A100 80GB: max ~300 concurrent requests!
└── 7.5x more concurrent requests = 7.5x more throughput
```

## Continuous Batching — Deep Dive

### The Problem: Static Batching

```
Static batching (naive):
├── Wait for N requests to form a batch
├── Process ALL requests in batch together
├── Batch completes when LONGEST request finishes
├── Short requests done early → GPU idle waiting for long one!

Timeline (static batch of 4 requests):
Req A (10 tokens):  [generate.......][idle......................]
Req B (50 tokens):  [generate..................................]
Req C (20 tokens):  [generate...........][idle.................]
Req D (5 tokens):   [generate...][idle.........................]
                                  ↑ All wait for B to finish!

GPU utilization: ~25% (3 of 4 requests idle most of the time)
```

### Continuous Batching Solution

```
Continuous batching (vLLM):
├── Requests added to batch AS THEY ARRIVE
├── Requests removed from batch AS THEY FINISH
├── GPU never idle — always processing active requests
├── New requests fill spots of completed ones immediately

Timeline (continuous batching):
Req A (10 tokens):  [gen][done] → slot freed
Req B (50 tokens):  [generating................................]
Req C (20 tokens):  [generating....][done] → slot freed
Req D (5 tokens):   [gen][done] → slot freed
Req E (new!):            [fills A's slot][generating...]
Req F (new!):                  [fills D's slot][gen...]
Req G (new!):                        [fills C's slot]

GPU utilization: ~95% (always full)
Throughput: 3-5x higher than static batching
```

### Iteration-Level Scheduling

```
Each "iteration" = one token generated for ALL active requests

Iteration 1: [A-token1, B-token1, C-token1, D-token1] → all 4 active
Iteration 2: [A-token2, B-token2, C-token2, D-token2]
...
Iteration 5: [A-done!, B-token5, C-token5, D-done!]  → A,D removed
Iteration 6: [E-token1, B-token6, C-token6, F-token1] → E,F added!
...
Iteration 10: [E-token5, B-token10, C-done!, F-token5]
Iteration 11: [E-token6, B-token11, G-token1, F-token6] → G fills C's spot

Key: Scheduler runs EVERY iteration (every token generation step)
```

## Running vLLM

### Basic Deployment

```bash
# Install
pip install vllm

# Start server (OpenAI-compatible API)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --dtype bfloat16

# Test
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Docker Deployment

```bash
# Official vLLM Docker image
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model meta-llama/Llama-2-7b-chat-hf \
  --tensor-parallel-size 1
```

### Multi-GPU (Tensor Parallelism)

```bash
# 70B model on 4 GPUs (tensor parallel)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --enforce-eager  # Disable CUDA graphs for debugging
```

### Advanced Configuration

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --max-num-batched-tokens 8192 \    # Max tokens in one batch
  --max-num-seqs 256 \               # Max concurrent sequences
  --dtype bfloat16 \
  --quantization awq \               # Use AWQ quantized model
  --enable-prefix-caching \          # Cache shared prefixes
  --swap-space 4 \                   # GB of CPU swap for preempted sequences
  --block-size 16 \                  # Tokens per KV-cache block
  --seed 42
```

## Key Features

### Prefix Caching
```
Scenario: 100 users ask questions with SAME system prompt (2000 tokens)

Without prefix caching:
  Each request: encode 2000 system tokens + user query
  100 requests = 200,000 tokens of redundant computation!

With prefix caching:
  First request: encode system prompt → cache KV-cache blocks
  Subsequent requests: reuse cached blocks (0 computation for system prompt!)
  Speedup: TTFT (Time to First Token) drops by 80%+

Enable: --enable-prefix-caching
```

### Speculative Decoding
```
Problem: LLM generates 1 token at a time (sequential, slow)
Solution: Use small "draft" model to predict N tokens, verify with large model

Draft model (7B): predicts tokens [A, B, C, D, E] (fast, ~70% accurate)
Target model (70B): verifies in ONE forward pass
  - Accepts [A, B, C] (correct predictions)
  - Rejects [D, E] (wrong)
  - Generates correct token for position 4
Result: 3 tokens generated in time of 1 target forward pass!

Enable: --speculative-model meta-llama/Llama-2-7b-chat-hf --num-speculative-tokens 5
(when serving a 70B model, use 7B as draft)
```

### Disaggregated Prefill & Decode
```
Problem: Prefill (processing input) is compute-bound,
         Decode (generating tokens) is memory-bound.
         They compete for same GPU resources.

Solution: Separate prefill and decode onto different GPU pools

Prefill pool (compute-optimized):
├── High SM utilization
├── Processes input prompts
└── Short, bursty

Decode pool (memory-optimized):
├── High memory bandwidth utilization
├── Generates tokens one at a time
└── Long-running

Benefit: Each pool optimized for its workload = higher overall throughput
```

## vLLM vs Others

```
Feature            │ vLLM        │ TGI         │ Triton+TRT-LLM │ Native HF
───────────────────┼─────────────┼─────────────┼─────────────────┼──────────
PagedAttention     │ ✓ (native)  │ ✓ (adopted) │ ✓ (in TRT-LLM)  │ ✗
Continuous Batch   │ ✓           │ ✓           │ ✓               │ ✗
Tensor Parallel    │ ✓ (easy)    │ ✓           │ ✓               │ Manual
Quantization       │ AWQ,GPTQ,FP8│ AWQ,GPTQ,BnB│ FP8,INT4,INT8  │ BitsAndBytes
Prefix Caching     │ ✓           │ ✗           │ ✓               │ ✗
Speculative Decode │ ✓           │ ✓           │ ✓               │ ✗
OpenAI API         │ ✓ (native)  │ ✓           │ Via wrapper     │ ✗
Throughput         │ ⭐⭐⭐⭐⭐     │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐⭐         │ ⭐⭐
Ease of use        │ ⭐⭐⭐⭐⭐     │ ⭐⭐⭐⭐⭐     │ ⭐⭐⭐            │ ⭐⭐⭐⭐⭐
Multi-modal        │ ✓           │ Limited     │ ✓               │ ✓
Production ready   │ ✓           │ ✓           │ ✓               │ ✗
```

## Performance Tuning

```
Tuning Guide:

1. GPU Memory Utilization:
   --gpu-memory-utilization 0.90  (default)
   ├── Higher = more KV-cache space = more concurrent requests
   ├── Too high = OOM risk
   └── Sweet spot: 0.85-0.95

2. Max Batch Size:
   --max-num-seqs 256
   ├── More concurrent requests = higher throughput
   ├── But: More requests = less KV-cache per request
   └── Balance based on your avg sequence length

3. Block Size:
   --block-size 16  (default)
   ├── Smaller blocks = less internal fragmentation
   ├── Larger blocks = less metadata overhead
   └── 16 is good default, try 32 for very long sequences

4. Swap Space:
   --swap-space 4  (GB)
   ├── When GPU memory full → preempt requests to CPU
   ├── Preempted requests resume when space available
   └── Prevents request failures at high load

5. dtype:
   --dtype bfloat16  (recommended for training-like precision)
   --dtype float16   (slightly more precision for some models)
   --dtype auto      (use model's native dtype)
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "meta-llama/Llama-2-7b-chat-hf"
        - "--tensor-parallel-size"
        - "1"
        - "--gpu-memory-utilization"
        - "0.90"
        - "--max-model-len"
        - "4096"
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 8Gi
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

## Interview Questions

1. **Q: PagedAttention se memory efficiency kaise improve hoti hai? Explain with numbers.**
   A: LLaMA-13B, max_seq=2048. Without paging: each request pre-allocates 1.6GB (full 2048 tokens). With paging: only actual tokens use memory (avg 256 tokens = 200MB). Result: 8x more concurrent requests on same GPU. Internal fragmentation drops from 60-80% to <4% (only last block partial).

2. **Q: Continuous batching vs static batching — throughput mein kitna farak?**
   A: Static: batch waits for longest request → GPU idle 60-75% time. Continuous: requests added/removed per iteration → GPU utilized 90%+. Real-world: 3-5x throughput improvement for mixed-length workloads. More variance in request lengths = more benefit from continuous batching.

3. **Q: vLLM pe 70B model serve karna hai 4× A100 pe. Config kya hogi?**
   A: `--tensor-parallel-size 4 --gpu-memory-utilization 0.95 --dtype bfloat16 --max-model-len 4096 --max-num-seqs 128 --enable-prefix-caching`. For 70B in BF16: ~140GB model weights across 4 GPUs (35GB each), leaving ~45GB per GPU for KV-cache. Can handle ~100-128 concurrent requests with avg 512 token sequences.

4. **Q: vLLM mein request preemption kab hota hai aur kaise kaam karta hai?**
   A: When GPU memory full and new high-priority request arrives. Steps: (1) Scheduler identifies lowest-priority active request, (2) Its KV-cache blocks are swapped to CPU memory (--swap-space), (3) New request gets GPU blocks, (4) When space available, preempted request's blocks swapped back from CPU, (5) Generation resumes from where it paused. No re-computation needed!
