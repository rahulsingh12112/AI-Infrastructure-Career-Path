# 02 — NVIDIA Triton Inference Server

## What is Triton?

Production-grade inference server from NVIDIA. Supports ALL frameworks, all optimizations.

```
Triton = Universal model serving platform:
├── Supports: PyTorch, TensorFlow, ONNX, TensorRT, Python, vLLM
├── Dynamic batching (automatic request batching)
├── Model ensemble (chain multiple models)
├── Concurrent model execution
├── GPU/CPU inference
├── Metrics (Prometheus-native)
├── Model versioning & hot-reload
└── Used by: NVIDIA, Microsoft, Amazon, most enterprises
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Triton Inference Server                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Client (gRPC/HTTP)                                          │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────────────────────────────────────┐       │
│  │              Request Handler                       │       │
│  │  ├── HTTP/REST endpoint (:8000)                   │       │
│  │  ├── gRPC endpoint (:8001)                        │       │
│  │  └── Metrics endpoint (:8002)                     │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │              Scheduler                             │       │
│  │  ├── Dynamic Batching Scheduler                   │       │
│  │  ├── Sequence Batching Scheduler                  │       │
│  │  └── Ensemble Scheduler                           │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │              Backend Framework                     │       │
│  │  ┌─────────┐ ┌──────────┐ ┌─────────┐           │       │
│  │  │TensorRT │ │  PyTorch │ │  ONNX   │           │       │
│  │  │Backend  │ │  Backend │ │ Runtime │           │       │
│  │  └─────────┘ └──────────┘ └─────────┘           │       │
│  │  ┌─────────┐ ┌──────────┐ ┌─────────┐           │       │
│  │  │TensorRT-│ │  Python  │ │  vLLM   │           │       │
│  │  │LLM      │ │  Backend │ │ Backend │           │       │
│  │  └─────────┘ └──────────┘ └─────────┘           │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  Model Repository (S3/local/GCS):                            │
│  models/                                                     │
│  ├── model_a/                                                │
│  │   ├── config.pbtxt                                        │
│  │   ├── 1/ (version 1)                                     │
│  │   │   └── model.plan (TensorRT engine)                   │
│  │   └── 2/ (version 2)                                     │
│  │       └── model.plan                                      │
│  └── model_b/                                                │
│      ├── config.pbtxt                                        │
│      └── 1/                                                  │
│          └── model.pt (PyTorch)                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Model Repository Structure

```bash
models/
├── bert-classifier/
│   ├── config.pbtxt              # Model configuration
│   ├── 1/                        # Version 1
│   │   └── model.onnx           # ONNX model file
│   └── 2/                        # Version 2 (latest)
│       └── model.onnx
│
├── llm-llama/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py              # Python backend (vLLM)
│
├── preprocessing/
│   ├── config.pbtxt
│   └── 1/
│       └── model.py              # Python preprocessing
│
└── ensemble_pipeline/
    └── config.pbtxt              # Chains preprocessing → model
```

## Model Configuration (config.pbtxt)

### Basic PyTorch Model
```protobuf
name: "bert-classifier"
platform: "pytorch_libtorch"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [512]          # Sequence length
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [512]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [2]            # Binary classification
  }
]

# Dynamic batching
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 100000    # Wait up to 100ms for batch
}

# Instance group (GPU assignment)
instance_group [
  {
    count: 2             # 2 instances of this model
    kind: KIND_GPU
    gpus: [0]            # On GPU 0
  }
]

# Version policy
version_policy: { latest { num_versions: 2 } }
```

### TensorRT-LLM for LLMs
```protobuf
name: "llama-7b"
backend: "tensorrtllm"
max_batch_size: 64

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [-1]           # Dynamic sequence length
  },
  {
    name: "input_lengths"
    data_type: TYPE_INT32
    dims: [1]
    reshape: { shape: [] }
  },
  {
    name: "request_output_len"
    data_type: TYPE_INT32
    dims: [1]
    reshape: { shape: [] }
  }
]

output [
  {
    name: "output_ids"
    data_type: TYPE_INT32
    dims: [-1, -1]
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [0, 1, 2, 3]   # Tensor parallel across 4 GPUs
  }
]

parameters: {
  key: "max_tokens_in_paged_kv_cache"
  value: { string_value: "8192" }
}
parameters: {
  key: "batch_scheduler_policy"
  value: { string_value: "guaranteed_no_evict" }
}
```

### Python Backend (Custom Logic)
```protobuf
name: "custom-preprocessor"
backend: "python"
max_batch_size: 64

input [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [1]
  }
]

output [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [512]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [512]
  }
]

instance_group [
  {
    count: 2
    kind: KIND_CPU          # Preprocessing on CPU
  }
]
```

### Python Backend Script (model.py)
```python
import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import AutoTokenizer
import json

class TritonPythonModel:
    def initialize(self, args):
        """Called once when model loads"""
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = 512
    
    def execute(self, requests):
        """Called for each batch of requests"""
        responses = []
        
        for request in requests:
            # Get input
            text = pb_utils.get_input_tensor_by_name(request, "text")
            text_str = text.as_numpy()[0][0].decode("utf-8")
            
            # Tokenize
            encoded = self.tokenizer(
                text_str,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            
            # Create output tensors
            input_ids = pb_utils.Tensor("input_ids", encoded["input_ids"].astype(np.int64))
            attention_mask = pb_utils.Tensor("attention_mask", encoded["attention_mask"].astype(np.int64))
            
            responses.append(pb_utils.InferenceResponse([input_ids, attention_mask]))
        
        return responses
    
    def finalize(self):
        """Called when model unloads"""
        pass
```

## Model Ensemble (Pipeline)

```protobuf
# ensemble_pipeline/config.pbtxt
name: "ensemble_pipeline"
platform: "ensemble"
max_batch_size: 32

input [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [1]
  }
]

output [
  {
    name: "classification"
    data_type: TYPE_FP32
    dims: [2]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "custom-preprocessor"
      model_version: -1
      input_map {
        key: "text"
        value: "text"
      }
      output_map {
        key: "input_ids"
        value: "input_ids"
      }
      output_map {
        key: "attention_mask"
        value: "attention_mask"
      }
    },
    {
      model_name: "bert-classifier"
      model_version: -1
      input_map {
        key: "input_ids"
        value: "input_ids"
      }
      input_map {
        key: "attention_mask"
        value: "attention_mask"
      }
      output_map {
        key: "logits"
        value: "classification"
      }
    }
  ]
}
```

## Running Triton

```bash
# Docker (recommended)
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# Verify
curl localhost:8000/v2/health/ready
# {"ready":true}

# Model status
curl localhost:8000/v2/models/bert-classifier
# Shows model metadata, versions, state

# Metrics (Prometheus format)
curl localhost:8002/metrics
```

## Client Libraries

```python
import tritonclient.grpc as grpcclient
import numpy as np

# Connect
client = grpcclient.InferenceServerClient(url="localhost:8001")

# Check health
print(client.is_server_ready())
print(client.is_model_ready("bert-classifier"))

# Inference
input_ids = grpcclient.InferInput("input_ids", [1, 512], "INT64")
input_ids.set_data_from_numpy(np.random.randint(0, 30000, (1, 512)).astype(np.int64))

attention_mask = grpcclient.InferInput("attention_mask", [1, 512], "INT64")
attention_mask.set_data_from_numpy(np.ones((1, 512)).astype(np.int64))

result = client.infer(
    model_name="bert-classifier",
    inputs=[input_ids, attention_mask],
    outputs=[grpcclient.InferRequestedOutput("logits")],
)

logits = result.as_numpy("logits")
print(f"Prediction: {logits}")
```

## Dynamic Batching Performance

```
Without batching:
  Request 1: [process] → 10ms
  Request 2: [      ][process] → 20ms total
  Request 3: [            ][process] → 30ms total
  Throughput: 100 req/sec

With dynamic batching (batch_size=8, max_delay=50ms):
  Requests 1-8: [batch process] → 15ms (not 8×10ms!)
  GPU processes 8 requests almost as fast as 1
  Throughput: 500+ req/sec

Why? GPU parallelism: matrix multiply for batch=8 is barely slower than batch=1
```

## Key Triton Features for Production

```
1. Model Hot-Reload:
   - Update model files in repository → Triton auto-reloads
   - Zero-downtime model updates
   - Version rollback (keep old versions)

2. Rate Limiting:
   - Queue size limits per model
   - Request timeout configuration
   - Protects against overload

3. Custom Metrics:
   - Request count, latency histograms
   - Queue size, batch size distribution
   - GPU utilization per model
   - Export to Prometheus/Grafana

4. Multi-GPU Model Placement:
   - Different models on different GPUs
   - Same model replicated across GPUs
   - Tensor parallelism for large models

5. Shared Memory:
   - Zero-copy data transfer between client and server
   - Critical for high-throughput scenarios
```

## Interview Questions

1. **Q: Triton vs vLLM — kab kya use karoge?**
   A: vLLM: Pure LLM serving (best PagedAttention, continuous batching, easier setup). Triton: Multi-framework serving (BERT + ResNet + LLM on same server), enterprise features (versioning, ensemble, metrics), or when using TensorRT-LLM for maximum GPU performance. For LLM-only: vLLM. For mixed models in production: Triton.

2. **Q: Dynamic batching configure kaise karoge 50ms latency SLA ke saath?**
   A: Set `max_queue_delay_microseconds: 30000` (30ms max wait). This leaves 20ms for actual inference. Set `preferred_batch_size: [4, 8, 16]` — server batches up to 16 requests within 30ms window. If fewer arrive, serves smaller batch immediately. Monitor p99 latency and adjust.

3. **Q: Model ensemble mein preprocessing slow hai. Kaise optimize?**
   A: (1) Python backend preprocessing on CPU (instance_group: KIND_CPU), (2) Increase instance count (count: 4 = 4 parallel preprocessors), (3) Move heavy preprocessing offline (pre-tokenize, store in cache), (4) Use C++ backend for perf-critical preprocessing, (5) Async pipeline — don't block GPU while preprocessing.

4. **Q: Triton mein 3 models hai same GPU pe. One model gets all traffic, others idle. Problem?**
   A: Default: models share GPU fairly. But bursty model can starve others. Fix: (1) Set priority_levels in config, (2) Use rate_limiter to cap requests per model, (3) Pin models to different GPUs (instance_group with specific gpu ids), (4) Use model-level queue size limits.
