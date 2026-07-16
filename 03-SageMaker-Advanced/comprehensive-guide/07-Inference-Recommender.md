# 07 — Inference Recommender & Optimization

## What is Inference Recommender?

Automatically benchmarks your model on different instance types to find the best price/performance.

```
Without Recommender:
├── "Which instance type for my model?" → Guess and pray
├── Try p4d.24xlarge ($32/hr) → works but expensive
├── Try g5.xlarge ($1/hr) → works! But is it fast enough?
├── Manual benchmarking = days of work
└── Wrong choice = overpay or under-perform

With Recommender:
├── Provide model → SageMaker tests on multiple instances
├── Get: latency, throughput, cost per 1M requests
├── Choose optimal instance in minutes, not days
└── Data-driven decision
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│              Inference Recommender Flow                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. You provide:                                             │
│     ├── Model artifact (S3)                                  │
│     ├── Sample payload (representative input)                │
│     └── Constraints (max latency, budget)                    │
│                                                              │
│  2. SageMaker runs:                                          │
│     ├── Deploys model on 5-10 instance types                │
│     ├── Sends benchmark traffic                              │
│     ├── Measures: latency (p50, p90, p99), throughput, cost │
│     └── Runs for ~45 minutes per instance                   │
│                                                              │
│  3. You get:                                                 │
│     ├── Ranked recommendations                               │
│     ├── Cost per inference                                   │
│     ├── Latency at different concurrency levels              │
│     └── Optimal instance + configuration                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Using Inference Recommender

### Default Job (Quick Recommendation)

```python
import sagemaker
from sagemaker.inference_recommender import InferenceRecommender

# Register model in Model Registry first (required)
model_package_arn = "arn:aws:sagemaker:us-east-1:123456:model-package/my-model/1"

# Run default recommendation job
recommender = InferenceRecommender(
    role=role,
    sagemaker_session=session,
)

# Default job: Tests on AWS-selected instance types
default_job = recommender.default(
    model_package_arn=model_package_arn,
    model_name="my-llm-model",
    framework="PYTORCH",
    framework_version="2.1",
    sample_payload_url="s3://my-bucket/sample-payload/payload.json",
    supported_content_types=["application/json"],
    job_name="my-inference-rec-job",
)

# Wait for results (~45 min)
default_job.wait()

# Get recommendations
results = default_job.describe()
for rec in results["InferenceRecommendations"]:
    print(f"""
    Instance: {rec['EndpointConfiguration']['InstanceType']}
    Latency P50: {rec['Metrics']['InferenceLatency']}ms
    Throughput: {rec['Metrics']['MaxInvocations']}/min
    Cost/hr: ${rec['Metrics']['CostPerHour']}
    Cost/1M requests: ${rec['Metrics']['CostPerInference'] * 1000000}
    """)
```

### Advanced Job (Custom Benchmarking)

```python
# Advanced: You specify exactly which instances to test
advanced_job = recommender.advanced(
    model_package_arn=model_package_arn,
    model_name="my-llm-model",
    framework="PYTORCH",
    sample_payload_url="s3://my-bucket/payload/",
    
    # Specific instances to test
    instance_types=[
        "ml.g5.xlarge",        # 1× A10G, 24GB, $1.00/hr
        "ml.g5.2xlarge",       # 1× A10G, 24GB + more CPU, $1.21/hr
        "ml.g5.4xlarge",       # 1× A10G, 24GB + lots CPU, $1.62/hr
        "ml.p4d.24xlarge",     # 8× A100, 80GB each, $32.77/hr
        "ml.inf2.xlarge",      # Inferentia2, $0.76/hr
    ],
    
    # Traffic patterns to simulate
    traffic_patterns=[
        {"TrafficType": "PHASES", "Phases": [
            {"InitialNumberOfUsers": 1, "SpawnRate": 1, "DurationInSeconds": 120},
            {"InitialNumberOfUsers": 10, "SpawnRate": 2, "DurationInSeconds": 300},
            {"InitialNumberOfUsers": 50, "SpawnRate": 5, "DurationInSeconds": 300},
        ]},
    ],
    
    # Constraints
    max_latency=100,          # Max 100ms p99 latency
    max_cost_per_hour=5.0,    # Max $5/hr budget
    
    # Resource limits
    max_tests=10,             # Max 10 benchmark runs
    
    job_name="advanced-rec-job",
)
```

### Interpret Results

```
Example Output:

Rank│ Instance       │ Latency P99 │ Throughput │ Cost/hr │ Cost/1M req
────┼────────────────┼─────────────┼────────────┼─────────┼────────────
 1  │ ml.g5.2xlarge  │ 45ms        │ 800/min   │ $1.21   │ $25.21
 2  │ ml.g5.xlarge   │ 52ms        │ 650/min   │ $1.00   │ $25.64
 3  │ ml.inf2.xlarge │ 38ms        │ 900/min   │ $0.76   │ $14.07  ★
 4  │ ml.g5.4xlarge  │ 42ms        │ 820/min   │ $1.62   │ $32.93
 5  │ ml.p4d.24xlarge│ 12ms        │ 5000/min  │ $32.77  │ $109.23

Analysis:
├── Cheapest per request: ml.inf2.xlarge ($14/1M) ← Best if model supported
├── Best latency: ml.p4d.24xlarge (12ms) ← Overkill for this model
├── Best balance: ml.g5.2xlarge (45ms, $25/1M) ← Good default
└── p4d way too expensive for single model inference
```

## Inference Optimization Techniques

### 1. Model Compilation (TensorRT / Neuron)

```python
# TensorRT: Optimize PyTorch model for NVIDIA GPUs
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    model_data="s3://bucket/model.tar.gz",
    role=role,
    framework_version="2.1",
    py_version="py310",
    env={
        "SAGEMAKER_PROGRAM": "inference.py",
        "TS_MAX_BATCH_SIZE": "32",        # Dynamic batching
        "TS_MAX_BATCH_DELAY": "100",       # Wait up to 100ms to form batch
    },
)

# For Inferentia2 (AWS Neuron):
from sagemaker.pytorch import PyTorchModel

neuron_model = PyTorchModel(
    model_data="s3://bucket/compiled-neuron-model.tar.gz",
    role=role,
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference-neuronx:2.1",
    env={
        "NEURON_RT_NUM_CORES": "2",       # Use 2 Neuron cores
    },
)
```

### 2. Quantization

```python
# Reduce model precision: FP32 → FP16 → INT8
# FP32: 4 bytes per param (baseline)
# FP16: 2 bytes per param (2x smaller, ~same accuracy)
# INT8: 1 byte per param (4x smaller, slight accuracy loss)

# Example with GPTQ quantization:
from transformers import AutoModelForCausalLM, GPTQConfig

quantization_config = GPTQConfig(
    bits=4,                    # 4-bit quantization
    dataset="wikitext2",       # Calibration dataset
    tokenizer=tokenizer,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto",
)
# 7B model: FP16 = 14GB → INT4 = 3.5GB (fits on smaller GPU!)
```

### 3. Dynamic Batching

```python
# Collect multiple requests → process as batch → return individual results
# Increases GPU utilization dramatically

# inference.py (custom handler)
import torch
from ts.torch_handler.base_handler import BaseHandler

class BatchHandler(BaseHandler):
    def __init__(self):
        self.batch_size = 32
        self.max_batch_delay_ms = 100  # Wait max 100ms for batch
    
    def handle(self, data, context):
        # 'data' is a list of inputs (batched by TorchServe)
        inputs = [self.preprocess(d) for d in data]
        
        # Batch forward pass (1 GPU call for N inputs)
        batch_tensor = torch.stack(inputs).cuda()
        with torch.no_grad():
            outputs = self.model(batch_tensor)
        
        # Return individual results
        return [self.postprocess(o) for o in outputs]

# TorchServe config (model-config.yaml):
# batchSize: 32
# maxBatchDelay: 100
# maxWorkers: 4
```

### 4. Model Caching & Warm-up

```python
# Pre-load model on endpoint start (avoid first-request latency)

# inference.py
import torch
import os

def model_fn(model_dir):
    """Called once when endpoint starts"""
    model = torch.load(os.path.join(model_dir, "model.pt"))
    model.cuda()
    model.eval()
    
    # Warm-up: Run dummy inference to compile CUDA kernels
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        _ = model(dummy_input)  # First run compiles, subsequent are fast
    
    return model
```

### 5. Inference Pipeline Optimization

```
Optimization Checklist:
├── Model level:
│   ├── Quantization (FP16/INT8/INT4)
│   ├── Pruning (remove unimportant weights)
│   ├── Knowledge distillation (smaller model, same quality)
│   ├── TensorRT compilation (GPU kernel optimization)
│   └── ONNX Runtime (cross-platform optimization)
│
├── Serving level:
│   ├── Dynamic batching (batch multiple requests)
│   ├── Model warmup (avoid cold start)
│   ├── Async inference (non-blocking for long jobs)
│   ├── Response caching (same input = cached output)
│   └── Connection pooling
│
├── Infrastructure level:
│   ├── Right-size instance (don't overpay)
│   ├── Auto-scaling (scale with demand)
│   ├── Multi-model endpoints (share resources)
│   ├── Spot instances (for batch inference)
│   └── Edge deployment (reduce network latency)
│
└── Data level:
    ├── Input validation (reject bad requests early)
    ├── Payload compression (smaller network transfer)
    ├── Feature store (pre-computed features)
    └── Streaming response (start sending before full output)
```

## SageMaker Inference Optimization Tools

```
Tool                    │ What it does                    │ Savings
────────────────────────┼─────────────────────────────────┼──────────
Inference Recommender   │ Find best instance type         │ 30-70% cost
Neo Compilation         │ Optimize for target hardware    │ 25-50% latency
Serverless Inference    │ Scale to zero                   │ 90% for bursty
Async Inference         │ Queue long-running inference    │ Better throughput
Batch Transform         │ Bulk offline inference          │ 60% vs real-time
Multi-Model Endpoint    │ Share instance across models    │ 80-95% for many models
```

## Interview Questions

1. **Q: New model deploy karna hai. Instance type kaise choose karoge?**
   A: (1) Run Inference Recommender with sample payloads, (2) Define constraints (max latency, budget), (3) Test 5-7 instance types (g5 series + inf2 + p4d), (4) Compare cost-per-1M-requests (not just $/hr), (5) Consider: GPU util at target throughput (should be 60-80%, not 100%). Usually g5.xlarge is sweet spot for single-model inference.

2. **Q: Model latency p99 = 500ms hai, requirement 100ms hai. Kaise optimize karoge?**
   A: Progressive optimization: (1) Quantize to FP16 (20-30% speedup, no accuracy loss), (2) TensorRT compilation (30-50% speedup), (3) Dynamic batching OFF if latency-sensitive (batching adds wait time), (4) Larger instance with faster GPU (A100 vs A10G), (5) If still not enough → model distillation (smaller model). Profile first to find bottleneck (CPU preprocess? GPU compute? Network?).

3. **Q: Inference cost $50K/month hai. Kaise reduce karoge?**
   A: (1) Inference Recommender → find cheaper instances, (2) MME for low-traffic models (biggest savings), (3) Auto-scaling (don't keep instances for peak during off-peak), (4) Serverless inference for bursty workloads (pay per request), (5) Quantization (smaller model = smaller instance), (6) Spot for batch inference, (7) Caching common responses (reduce actual GPU calls).

4. **Q: SageMaker Serverless Inference kab use karna chahiye?**
   A: When: (1) Traffic unpredictable/bursty (0 to 100 requests randomly), (2) Can tolerate cold starts (1-3 sec), (3) Cost matters more than latency, (4) Model is small (<5GB). When NOT: (1) Consistent high traffic, (2) Low-latency requirement (<100ms), (3) GPU needed (serverless is CPU only currently), (4) Large models.
