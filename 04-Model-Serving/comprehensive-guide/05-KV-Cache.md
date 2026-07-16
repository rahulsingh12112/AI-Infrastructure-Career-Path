# 05 — KV-Cache Optimization

## What is KV-Cache?

During text generation, the model computes Key and Value tensors for every token. Re-computing them every step is wasteful → cache them!

```
Without KV-Cache:
  Generate token 100:
  → Re-process ALL previous 99 tokens (attention over full sequence)
  → O(n²) compute per token!
  → Extremely slow for long sequences

With KV-Cache:
  Generate token 100:
  → Only compute new token's Q, use cached K,V from tokens 1-99
  → O(n) compute per token
  → Fast, but uses LOTS of memory!
```

## KV-Cache Memory Calculation

```
KV-Cache size per token per layer:
= 2 (K + V) × hidden_size × num_heads × dtype_size

For LLaMA-7B (32 layers, 4096 hidden, 32 heads, FP16):
Per token: 2 × 4096 × 2 bytes × 32 layers = 524 KB per token!

For 2048 token sequence:
= 524 KB × 2048 = 1.07 GB per request!

For LLaMA-70B (80 layers, 8192 hidden, 64 heads, FP16):
Per token: 2 × 8192 × 2 bytes × 80 layers = 2.62 MB per token!
For 2048 tokens: 2.62 MB × 2048 = 5.37 GB per request!

Impact:
├── A100 80GB with LLaMA-70B:
│   Model weights: ~140GB (need 2 GPUs minimum)
│   Remaining for KV-cache per GPU: ~10GB
│   Max concurrent requests: 10GB / 5.37GB ≈ 2 requests!
│   Without optimization: Can only serve 2 users simultaneously!
└── This is why KV-cache optimization is CRITICAL for LLM serving
```

## KV-Cache Optimization Techniques

### 1. PagedAttention (covered in vLLM chapter)
```
Problem: Pre-allocated contiguous memory → fragmentation
Solution: Pages/blocks, allocated on-demand
Savings: 60-80% less memory waste → 3-5x more concurrent requests
```

### 2. Multi-Query Attention (MQA) & Grouped-Query Attention (GQA)

```
Standard Multi-Head Attention (MHA):
├── Each head has its own K, V, Q
├── 32 heads = 32 K matrices + 32 V matrices
├── KV-cache: 32 × 2 × hidden_per_head × seq_len
└── Used by: GPT-3, original LLaMA

Multi-Query Attention (MQA):
├── ALL heads share SAME K and V (only Q is per-head)
├── 32 Q heads, but only 1 K and 1 V
├── KV-cache: 1 × 2 × hidden_per_head × seq_len
├── 32x less KV-cache memory!
├── Slight quality loss
└── Used by: Falcon, PaLM, StarCoder

Grouped-Query Attention (GQA):
├── Compromise: Groups of heads share K,V
├── 32 Q heads, 8 KV groups (4 Q heads per KV group)
├── KV-cache: 8 × 2 × hidden_per_head × seq_len
├── 4x less KV-cache than MHA, better quality than MQA
└── Used by: LLaMA-2 70B, Mistral, Mixtral

Memory comparison (LLaMA-2-70B with GQA, 8 KV heads):
MHA would be: 64 × 2 × 128 × seq_len = 16384 × seq_len bytes
GQA (8 groups): 8 × 2 × 128 × seq_len = 2048 × seq_len bytes
= 8x less KV-cache! (70B can serve many more concurrent users)
```

### 3. KV-Cache Quantization

```
Idea: Store KV-cache in lower precision (FP8 or INT8)

Standard: KV-cache in FP16 (2 bytes per element)
Quantized: KV-cache in FP8 (1 byte) or INT4 (0.5 bytes)

Savings:
├── FP16 → FP8: 2x more concurrent requests
├── FP16 → INT4: 4x more concurrent requests
├── Quality impact: Minimal for KV-cache (it's intermediate state)
└── Supported: vLLM (FP8 KV-cache), TensorRT-LLM

# vLLM with KV-cache quantization:
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --kv-cache-dtype fp8           # FP8 KV-cache!

# Result: Same model, 2x more concurrent requests
```

### 4. Sliding Window Attention

```
Standard attention: Each token attends to ALL previous tokens
KV-cache grows: O(seq_len) per layer

Sliding Window (used by Mistral):
├── Each token only attends to last W tokens (window)
├── KV-cache fixed at: W × num_layers × hidden_size
├── Doesn't grow with sequence length!
├── W = 4096 for Mistral (still captures long-range via layer stacking)
└── Memory: CONSTANT regardless of sequence length

Example:
  Sequence length: 32768 tokens
  Without sliding window: KV-cache = 32768 × layers × hidden = HUGE
  With sliding window (W=4096): KV-cache = 4096 × layers × hidden = FIXED
  
  Memory saving at 32K tokens: 8x less!
```

### 5. Token Eviction / Attention Sink

```
Observation: In long sequences, attention concentrates on:
├── First few tokens ("attention sink" tokens)
├── Recent tokens (local context)
└── Middle tokens get very low attention → can be evicted!

StreamingLLM approach:
├── Keep: First 4 tokens (sink) + last 1024 tokens (recent)
├── Evict: Everything in between
├── Result: Constant memory regardless of total sequence length
├── Quality: Good for streaming/chat (local context matters most)
└── Bad for: Tasks requiring full context (summarization, RAG)

┌─────────────────────────────────────────────────────────┐
│ KV-Cache with Attention Sink:                            │
│                                                          │
│ [Sink: 4 tokens] [...evicted...] [Recent: 1024 tokens]  │
│ ├── Fixed size: 1028 tokens always                      │
│ ├── Can generate infinitely long text                   │
│ └── Never runs out of memory!                           │
└─────────────────────────────────────────────────────────┘
```

### 6. Prefix Caching

```
Scenario: System prompt is same for all users (2000 tokens)

Without prefix caching:
  User A: [encode system prompt (2000 tokens)] + [encode query]
  User B: [encode system prompt (2000 tokens)] + [encode query]
  User C: [encode system prompt (2000 tokens)] + [encode query]
  = 6000 tokens of redundant KV-cache computation!

With prefix caching:
  First user: [encode system prompt] → CACHE the KV-cache blocks
  User B: [reuse cached KV blocks] + [encode query only]
  User C: [reuse cached KV blocks] + [encode query only]
  = 2000 tokens computed ONCE, reused for everyone!

Savings:
├── TTFT (Time to First Token): 80% reduction for cached prefix
├── Memory: Shared blocks, not duplicated per request
├── Throughput: More requests processed (less prefill compute)
└── Enable in vLLM: --enable-prefix-caching
```

### 7. Speculative Decoding (KV-Cache Perspective)

```
How speculative decoding helps KV-cache:

Without speculative:
  Large model: Generate 1 token → add to KV-cache → 1 token → add → ...
  100 tokens = 100 forward passes of large model

With speculative (draft model):
  Draft model: Generate 5 tokens quickly (small KV-cache)
  Large model: Verify all 5 in ONE forward pass
  If 3 accepted: Generated 3 tokens in 1 large-model forward pass!
  
  KV-cache benefit: 
  ├── Large model's KV-cache grows by 3 tokens in 1 step (not 3 steps)
  ├── Less time spent on KV-cache management
  └── Higher throughput for same memory
```

## KV-Cache Memory Budget Planning

```
Planning for production:

Given: A100 80GB, LLaMA-2-7B (FP16)
├── Model weights: 14 GB
├── CUDA overhead: ~2 GB
├── Activation memory: ~2 GB
├── Available for KV-cache: 80 - 14 - 2 - 2 = 62 GB
│
├── KV-cache per token: 524 KB
├── Max tokens in cache: 62 GB / 524 KB ≈ 118,000 tokens
│
├── If avg request = 1024 tokens (input + output):
│   Max concurrent requests: 118,000 / 1024 ≈ 115 requests
│
├── If avg request = 4096 tokens:
│   Max concurrent requests: 118,000 / 4096 ≈ 28 requests
│
└── With FP8 KV-cache: Double all numbers above!

Planning formula:
max_concurrent = (GPU_mem - model_size - overhead) / (kv_per_token × avg_seq_len)
```

## Optimization Comparison

```
Optimization           │ Memory Saving │ Quality Impact │ Complexity
───────────────────────┼───────────────┼────────────────┼───────────
PagedAttention         │ 2-4x          │ None           │ Low (use vLLM)
GQA (model design)     │ 4-8x          │ Minimal        │ None (model has it)
KV-cache quantization  │ 2-4x          │ Minimal        │ Low
Sliding window         │ Varies        │ For long seqs  │ None (model has it)
Token eviction         │ Varies        │ Some loss      │ Medium
Prefix caching         │ Varies        │ None           │ Low (vLLM flag)
Speculative decoding   │ Indirect      │ None           │ Medium

Stack them: PagedAttention + GQA + FP8 KV-cache + prefix caching
= Maximum concurrent requests on same hardware
```

## Interview Questions

1. **Q: LLaMA-70B serve karna hai, max concurrent users kitne ho sakte hain A100 80GB × 2 pe?**
   A: 70B FP16 = 140GB across 2 GPUs (70GB each). Remaining per GPU: 10GB for KV-cache. 70B with GQA (8 KV heads): per token KV = 8 × 2 × 128 × 80 layers × 2 bytes = 327 KB. Per 2048 token seq = 670 MB. Max concurrent: 10GB / 670MB ≈ 15 per GPU × 2 = ~30 requests. With FP8 KV-cache: ~60 requests. With AWQ 4-bit (single GPU): model=35GB, KV-cache=45GB → ~67 requests on 1 GPU!

2. **Q: KV-cache memory badhti ja rahi hai, eventually OOM. Kaise handle karoge?**
   A: (1) PagedAttention with swap space (preempt to CPU), (2) Set max-model-len limit, (3) KV-cache quantization (FP8), (4) If model supports: sliding window attention, (5) Request timeout (kill long-running generations), (6) vLLM automatically preempts lowest-priority requests when memory full.

3. **Q: Prefix caching se TTFT 80% kam hua but throughput same hai. Kyun?**
   A: Prefix caching saves prefill compute (reduces TTFT) but doesn't help decode phase. Throughput is dominated by decode (memory-bandwidth bound). To improve throughput: (1) More concurrent requests (continuous batching), (2) KV-cache quantization (more room for requests), (3) Speculative decoding (more tokens per forward pass).

4. **Q: GQA model vs MHA model — serving pe kya impact?**
   A: GQA (like LLaMA-2-70B, 8 KV heads) uses 8x less KV-cache than equivalent MHA model. Direct impact: 8x more concurrent requests on same GPU. This is why modern LLMs use GQA — not just for training speed but critically for inference serving scalability. Choose GQA models for production serving whenever possible.
