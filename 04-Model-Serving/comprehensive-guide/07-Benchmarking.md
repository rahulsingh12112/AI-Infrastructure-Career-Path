# 07 — Benchmarking: Latency, Throughput, Cost

## Why Benchmark?

```
Without benchmarking:
├── "Model is slow" — how slow? compared to what?
├── "Need bigger GPU" — do you? or is the bottleneck elsewhere?
├── "We can handle 100 users" — can you really? At what latency?
└── Decisions based on guessing = overpaying or under-delivering

With proper benchmarking:
├── Exact throughput: 50 requests/sec at p99 < 2s
├── Cost: $0.002 per 1000 tokens
├── Capacity planning: Need 4 GPUs for 200 concurrent users
└── Optimization: Identified that data loading is bottleneck (not GPU)
```

## Key Metrics for LLM Inference

```
Metric                          │ Definition                              │ Target
────────────────────────────────┼─────────────────────────────────────────┼──────────
TTFT (Time to First Token)      │ Time from request to first output token │ < 500ms
TPOT (Time Per Output Token)    │ Time between consecutive tokens         │ < 50ms
TPS (Tokens Per Second)         │ Output tokens generated per second      │ > 30 t/s/user
Throughput (total)              │ Total tokens/sec across all requests    │ Maximize
Latency (end-to-end)           │ Full request → complete response time   │ < 10s (chat)
P50 / P99 latency              │ Median / tail latency                   │ P99 < 2× P50
Concurrent requests             │ Number of simultaneous requests         │ > 50
Queue depth                     │ Requests waiting to be processed       │ < 10
GPU utilization                 │ % of GPU compute used                  │ > 70%
KV-cache utilization            │ % of KV-cache memory used              │ 60-85%
Cost per 1M tokens              │ Dollar cost for generating 1M tokens   │ Minimize
```

### Metric Relationships

```
TTFT = Prefill time (processing input tokens)
     = Affected by: input length, batch size, GPU compute

TPOT = Decode time per token
     = Affected by: model size, memory bandwidth, batch size

End-to-end latency = TTFT + (output_tokens × TPOT)

Throughput = concurrent_requests × (1 / TPOT)
           = Affected by: batch size, memory for KV-cache

Trade-off: Latency ↔ Throughput
├── More batching = higher throughput BUT higher latency
├── Less batching = lower latency BUT lower throughput
└── Find the sweet spot for your SLA
```

## Benchmarking Tools

### 1. vLLM Benchmark Script
```bash
# Built-in vLLM benchmark
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf &

# Run benchmark
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model meta-llama/Llama-2-7b-chat-hf \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --request-rate 10 \           # 10 requests/sec
  --endpoint http://localhost:8000/v1/completions
```

### 2. Custom Benchmark Script
```python
import asyncio
import aiohttp
import time
import statistics
import json

async def send_request(session, url, prompt, max_tokens):
    """Send single request and measure timing"""
    payload = {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    
    start = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    async with session.post(url, json=payload) as response:
        async for line in response.content:
            if line:
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data:") and "[DONE]" not in decoded:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
    
    end = time.perf_counter()
    
    return {
        "ttft": (first_token_time - start) if first_token_time else None,
        "total_time": end - start,
        "tokens": token_count,
        "tpot": (end - first_token_time) / token_count if token_count > 1 else None,
    }

async def run_benchmark(url, num_requests, concurrency, max_tokens=100):
    """Run benchmark with specified concurrency"""
    prompts = ["Explain quantum computing in detail." for _ in range(num_requests)]
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    async def limited_request(session, prompt):
        async with semaphore:
            return await send_request(session, url, prompt, max_tokens)
    
    start_time = time.perf_counter()
    
    async with aiohttp.ClientSession() as session:
        tasks = [limited_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
    
    total_time = time.perf_counter() - start_time
    
    # Calculate statistics
    ttfts = [r["ttft"] for r in results if r["ttft"]]
    tpots = [r["tpot"] for r in results if r["tpot"]]
    total_tokens = sum(r["tokens"] for r in results)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Requests:         {num_requests}")
    print(f"Concurrency:      {concurrency}")
    print(f"Total time:       {total_time:.2f}s")
    print(f"Throughput:       {num_requests/total_time:.2f} req/s")
    print(f"Token throughput:  {total_tokens/total_time:.2f} tokens/s")
    print(f"")
    print(f"TTFT (Time to First Token):")
    print(f"  P50: {statistics.median(ttfts)*1000:.1f}ms")
    print(f"  P90: {sorted(ttfts)[int(0.9*len(ttfts))]*1000:.1f}ms")
    print(f"  P99: {sorted(ttfts)[int(0.99*len(ttfts))]*1000:.1f}ms")
    print(f"")
    print(f"TPOT (Time Per Output Token):")
    print(f"  P50: {statistics.median(tpots)*1000:.1f}ms")
    print(f"  P99: {sorted(tpots)[int(0.99*len(tpots))]*1000:.1f}ms")
    print(f"{'='*60}")

# Run
asyncio.run(run_benchmark(
    url="http://localhost:8000/v1/completions",
    num_requests=500,
    concurrency=50,
    max_tokens=100,
))
```

### 3. locust for Load Testing
```python
# locustfile.py
from locust import HttpUser, task, between
import json

class LLMUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def generate(self):
        self.client.post(
            "/v1/completions",
            json={
                "model": "meta-llama/Llama-2-7b-chat-hf",
                "prompt": "Explain machine learning in simple terms.",
                "max_tokens": 100,
                "temperature": 0.7,
            },
            headers={"Content-Type": "application/json"},
        )

# Run: locust -f locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 10
```

## Expected Performance (Reference Numbers)

```
LLaMA-2-7B on different hardware (vLLM, BF16):

Hardware      │ TTFT (128 in)│ TPOT    │ Throughput │ Concurrent │ Cost/1M tokens
──────────────┼──────────────┼─────────┼────────────┼────────────┼───────────────
A10G (24GB)   │ 80ms         │ 35ms    │ 800 t/s    │ 28         │ $0.45
A100 (80GB)   │ 40ms         │ 18ms    │ 2500 t/s   │ 90         │ $0.30
H100 (80GB)   │ 25ms         │ 12ms    │ 4000 t/s   │ 130        │ $0.25

LLaMA-2-70B (AWQ 4-bit on single A100):
──────────────┼──────────────┼─────────┼────────────┼────────────┼───────────────
A100 (80GB)   │ 200ms        │ 45ms    │ 600 t/s    │ 25         │ $1.20

LLaMA-2-70B (FP16 on 4× A100, TP=4):
──────────────┼──────────────┼─────────┼────────────┼────────────┼───────────────
4×A100 (320GB)│ 120ms        │ 30ms    │ 1500 t/s   │ 50         │ $1.80

Note: These are approximate. Actual numbers depend on:
├── Input/output length distribution
├── Batch size and traffic pattern
├── Model configuration (rope, GQA, etc.)
└── vLLM version and kernel optimizations
```

## Cost Analysis Framework

```python
def calculate_cost_per_token(
    instance_cost_per_hour: float,
    throughput_tokens_per_second: float,
) -> float:
    """Calculate cost per 1M output tokens"""
    tokens_per_hour = throughput_tokens_per_second * 3600
    cost_per_token = instance_cost_per_hour / tokens_per_hour
    cost_per_million = cost_per_token * 1_000_000
    return cost_per_million

# Examples:
# A10G ($1.00/hr), 800 tokens/sec
print(f"A10G: ${calculate_cost_per_token(1.00, 800):.3f} per 1M tokens")
# = $0.347 per 1M tokens

# A100 ($4.10/hr), 2500 tokens/sec
print(f"A100: ${calculate_cost_per_token(4.10, 2500):.3f} per 1M tokens")
# = $0.456 per 1M tokens

# H100 ($8.00/hr), 4000 tokens/sec  
print(f"H100: ${calculate_cost_per_token(8.00, 4000):.3f} per 1M tokens")
# = $0.556 per 1M tokens

# Insight: A10G is cheapest per token for 7B model!
# But A100 handles more concurrent users (better for latency SLA)
```

## Benchmarking Methodology

```
Proper benchmarking process:

1. WARMUP (critical!):
   ├── First 50-100 requests: Model compiling CUDA kernels, JIT
   ├── Discard warmup results
   └── Measure only after system reaches steady state

2. REALISTIC LOAD:
   ├── Use real traffic patterns (not constant rate)
   ├── Mix of short and long requests
   ├── Include system prompts (if your app uses them)
   └── ShareGPT dataset is good proxy for chat workloads

3. MEASURE CORRECTLY:
   ├── Client-side measurement (includes network)
   ├── Measure TTFT and TPOT separately
   ├── Report P50, P90, P99 (not just average!)
   ├── Run for 10+ minutes (steady state)
   └── Multiple runs, report confidence intervals

4. COMPARE FAIRLY:
   ├── Same model, same precision
   ├── Same input/output length distribution
   ├── Same max concurrent requests
   ├── Same hardware (GPU type, memory)
   └── Document ALL settings

5. COMMON MISTAKES:
   ├── ❌ Reporting average latency (hides tail)
   ├── ❌ Testing with 1 request at a time (no batching benefit)
   ├── ❌ Not warming up (includes compilation time)
   ├── ❌ Fixed input length (unrealistic)
   └── ❌ Ignoring throughput-latency trade-off
```

## Production Monitoring Dashboard

```
Essential panels for LLM inference monitoring:

Row 1: Traffic
├── Requests/sec (rate)
├── Active/queued requests (gauge)
└── Error rate (%)

Row 2: Latency
├── TTFT P50/P99 (time series)
├── TPOT P50/P99 (time series)
└── End-to-end latency distribution (histogram)

Row 3: Throughput
├── Tokens/sec generated (rate)
├── Batch size distribution (histogram)
└── Tokens per request (histogram)

Row 4: Resources
├── GPU utilization (%)
├── GPU memory / KV-cache usage (%)
└── CPU utilization (%)

Row 5: Cost
├── Cost per request (calculated)
├── Cost per 1M tokens (calculated)
├── Total hourly/daily cost (counter)
└── Cost per user/team (breakdown)
```

## Interview Questions

1. **Q: LLM endpoint benchmark karna hai. Kaise karoge systematically?**
   A: (1) Choose representative dataset (ShareGPT for chat), (2) Warmup 100 requests (discard), (3) Send at increasing rates (1, 5, 10, 20, 50 req/s), (4) Measure: TTFT, TPOT, throughput, error rate at each level, (5) Find saturation point (where P99 latency breaks SLA), (6) Report throughput at SLA-meeting load level. Run 10+ min per rate for stability.

2. **Q: Benchmark mein P99 latency 5x higher hai P50 se. Kyun?**
   A: Tail latency causes: (1) KV-cache full → request preempted/queued, (2) Long input request blocks batch (prefill takes time), (3) GC pause in Python, (4) Spot interruption/node issue, (5) Large batch forming (max_queue_delay). Fix: Set tighter queue limits, separate long/short requests, reduce max-model-len.

3. **Q: A100 pe 7B model serve kar rahe ho. Cost optimization kaise karoge?**
   A: A100 is overkill for 7B! Calculate: A100 ($4.10/hr) with 2500 t/s = $0.46/1M tokens. A10G ($1.00/hr) with 800 t/s = $0.35/1M tokens. Switch to A10G = 24% cheaper per token. If need more throughput: 4× A10G ($4/hr) = 3200 t/s = $0.35/1M. Better than 1× A100 in both throughput AND cost. Use A100 only for models that don't fit in 24GB.

4. **Q: Customer SLA: TTFT < 500ms, TPOT < 50ms. Capacity planning kaise karoge 1000 concurrent users ke liye?**
   A: (1) Benchmark single instance: At what concurrency does TTFT hit 500ms? Say 30 concurrent. (2) Need: 1000/30 ≈ 34 instances minimum. (3) Add 50% headroom for bursts: 50 instances. (4) With autoscaling: min=34, max=50. (5) Cost: 50 × g5.xlarge × $1/hr = $50/hr = $36K/month. (6) Optimize: Quantize model (more concurrent per instance), reduce to 25 instances → $18K/month.
