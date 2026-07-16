# 08 — Hands-On Labs (Model Serving)

## Prerequisites
```bash
pip install vllm transformers torch openai aiohttp locust
# GPU required (at least T4 or A10G)
# Docker installed
```

---

## Lab 1: Deploy vLLM and Benchmark

### Objective
Deploy LLaMA-2-7B with vLLM, benchmark throughput and latency.

```bash
# 1. Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --gpu-memory-utilization 0.90 \
  --max-model-len 2048 \
  --port 8000 &

# Wait for model to load (check logs)
sleep 60

# 2. Test single request
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-2-7b-chat-hf","prompt":"What is AI?","max_tokens":50}'

# 3. Benchmark with increasing concurrency
for CONCURRENT in 1 5 10 20 50; do
  echo "=== Concurrency: $CONCURRENT ==="
  python benchmark.py --concurrency $CONCURRENT --requests 100
done
```

### Verification
- [ ] vLLM starts and serves requests
- [ ] Can identify throughput at different concurrency levels
- [ ] Can identify P99 latency at saturation point

---

## Lab 2: TGI Deployment with Streaming

### Objective
Deploy model with TGI, implement streaming client.

```bash
# 1. Start TGI
docker run --gpus all --shm-size 1g -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-chat-hf \
  --max-input-length 1024 \
  --max-total-tokens 2048

# 2. Test non-streaming
curl localhost:8080/generate \
  -X POST -H 'Content-Type: application/json' \
  -d '{"inputs":"Explain deep learning","parameters":{"max_new_tokens":100}}'

# 3. Test streaming
curl localhost:8080/generate_stream \
  -X POST -H 'Content-Type: application/json' \
  -d '{"inputs":"Write a poem about AI","parameters":{"max_new_tokens":200}}'
```

### Python Streaming Client
```python
import requests, json

def stream_response(prompt):
    response = requests.post(
        "http://localhost:8080/generate_stream",
        json={"inputs": prompt, "parameters": {"max_new_tokens": 200}},
        stream=True,
    )
    for line in response.iter_lines():
        if line:
            data = line.decode("utf-8")
            if data.startswith("data:"):
                token = json.loads(data[5:])
                print(token["token"]["text"], end="", flush=True)
    print()

stream_response("Explain quantum computing step by step")
```

### Verification
- [ ] TGI serves responses correctly
- [ ] Streaming delivers tokens one by one
- [ ] Can observe TTFT vs total latency difference

---

## Lab 3: Quantize Model (AWQ) and Compare

### Objective
Quantize a model with AWQ, compare size, speed, and quality.

```python
# quantize_awq.py
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "./llama-7b-awq-4bit"

# Load
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config)

# Save
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f"Quantized model saved to {quant_path}")
```

```bash
# Compare sizes
du -sh /path/to/original-model/   # ~14 GB
du -sh ./llama-7b-awq-4bit/       # ~3.5 GB

# Serve quantized model
python -m vllm.entrypoints.openai.api_server \
  --model ./llama-7b-awq-4bit \
  --quantization awq \
  --gpu-memory-utilization 0.90 \
  --port 8001

# Benchmark both (original on 8000, quantized on 8001)
python benchmark.py --port 8000 --label "FP16"
python benchmark.py --port 8001 --label "AWQ-4bit"
```

### Verification
- [ ] Model quantized to ~25% of original size
- [ ] Quantized model serves correctly
- [ ] Speed improvement measurable (1.5-2x)
- [ ] Quality comparison (run same prompts, compare outputs)

---

## Lab 4: Triton Inference Server Setup

### Objective
Deploy a model on Triton with dynamic batching.

```bash
# 1. Prepare model repository
mkdir -p models/bert-classifier/1

# Export model to TorchScript or ONNX
python export_model.py  # Creates model.onnx

cp model.onnx models/bert-classifier/1/

# 2. Create config
cat > models/bert-classifier/config.pbtxt << 'EOF'
name: "bert-classifier"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  { name: "input_ids", data_type: TYPE_INT64, dims: [512] },
  { name: "attention_mask", data_type: TYPE_INT64, dims: [512] }
]
output [
  { name: "logits", data_type: TYPE_FP32, dims: [2] }
]
dynamic_batching {
  preferred_batch_size: [8, 16, 32]
  max_queue_delay_microseconds: 50000
}
instance_group [
  { count: 2, kind: KIND_GPU, gpus: [0] }
]
EOF

# 3. Start Triton
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models

# 4. Test with client
python triton_client.py
```

### Triton Client
```python
import tritonclient.http as httpclient
import numpy as np

client = httpclient.InferenceServerClient(url="localhost:8000")
assert client.is_server_ready()

# Send request
input_ids = httpclient.InferInput("input_ids", [1, 512], "INT64")
input_ids.set_data_from_numpy(np.ones((1, 512), dtype=np.int64))

attention_mask = httpclient.InferInput("attention_mask", [1, 512], "INT64")
attention_mask.set_data_from_numpy(np.ones((1, 512), dtype=np.int64))

result = client.infer("bert-classifier", [input_ids, attention_mask])
print(f"Output: {result.as_numpy('logits')}")
```

### Verification
- [ ] Triton starts with model loaded
- [ ] Dynamic batching active (check metrics)
- [ ] Client gets correct predictions
- [ ] Metrics available on :8002/metrics

---

## Lab 5: KV-Cache Optimization Comparison

### Objective
Measure impact of KV-cache optimizations on concurrent request capacity.

```bash
# Test 1: Standard (no optimization)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --port 8000 &

# Measure max concurrent at target latency
python benchmark.py --port 8000 --find-max-concurrent --target-ttft 500

# Test 2: With FP8 KV-cache
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --kv-cache-dtype fp8 \
  --port 8001 &

python benchmark.py --port 8001 --find-max-concurrent --target-ttft 500

# Test 3: With prefix caching (shared system prompt)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --enable-prefix-caching \
  --port 8002 &

python benchmark.py --port 8002 --find-max-concurrent --target-ttft 500 --system-prompt
```

### Verification
- [ ] FP8 KV-cache: ~2x more concurrent requests
- [ ] Prefix caching: Significant TTFT reduction for shared prefixes
- [ ] Document exact numbers for your hardware

---

## Lab 6: Autoscaling on Kubernetes

### Objective
Set up HPA with custom metrics for LLM inference.

```yaml
# 1. Deploy vLLM with Prometheus metrics
# (vLLM exposes /metrics endpoint)

# 2. Install Prometheus Adapter (for custom metrics)
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --set prometheus.url=http://prometheus-server.monitoring:80 \
  --set rules.custom[0].seriesQuery='vllm:num_requests_waiting' \
  --set rules.custom[0].metricsQuery='sum(vllm:num_requests_waiting)' \
  --set rules.custom[0].name.as='vllm_queue_size'

# 3. Apply HPA
kubectl apply -f hpa-vllm.yaml

# 4. Generate load
kubectl run loadgen --image=python:3.10 -- python /scripts/load_generator.py

# 5. Watch scaling
kubectl get hpa -w
kubectl get pods -w
```

### Verification
- [ ] HPA scales up when queue depth increases
- [ ] New pods start and join serving pool
- [ ] HPA scales down after load decreases
- [ ] Latency stays within SLA during scaling

---

## Lab 7: Full Production Comparison (vLLM vs TGI)

### Objective
Compare vLLM and TGI on same hardware, same model, same workload.

```bash
# Same model, same hardware, same benchmark

# Start vLLM
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --gpu-memory-utilization 0.90 --port 8000 &

# Start TGI (different port)
docker run --gpus '"device=1"' --shm-size 1g -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-chat-hf

# Run same benchmark on both
python benchmark.py --port 8000 --label "vLLM" --requests 500 --concurrency 30
python benchmark.py --port 8080 --label "TGI" --requests 500 --concurrency 30

# Compare:
# - TTFT P50/P99
# - TPOT P50/P99
# - Total throughput (tokens/sec)
# - Max concurrent before SLA breach
```

### Expected Results
```
Metric              │ vLLM          │ TGI
────────────────────┼───────────────┼──────────────
TTFT P50            │ ~80ms         │ ~90ms
TTFT P99            │ ~200ms        │ ~220ms
TPOT P50            │ ~30ms         │ ~32ms
Throughput          │ ~850 t/s      │ ~780 t/s
Max concurrent      │ ~90           │ ~80
```

### Verification
- [ ] Both serving correctly
- [ ] Benchmark results documented
- [ ] Clear winner identified for your use case
- [ ] Cost per token calculated for both

---

## Summary Checklist

- [ ] Lab 1: vLLM deployed, benchmarked at multiple concurrency levels
- [ ] Lab 2: TGI streaming working
- [ ] Lab 3: AWQ quantization done, speed/quality compared
- [ ] Lab 4: Triton with dynamic batching serving
- [ ] Lab 5: KV-cache optimizations measured
- [ ] Lab 6: Autoscaling on K8s working
- [ ] Lab 7: vLLM vs TGI comparison completed
