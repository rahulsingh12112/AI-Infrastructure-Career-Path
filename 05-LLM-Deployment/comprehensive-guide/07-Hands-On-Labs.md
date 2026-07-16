# 07 — Hands-On Labs (LLM Deployment)

## Lab 1: Deploy on Bedrock

```python
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Invoke Claude
response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    }),
)
print(json.loads(response["body"].read())["content"][0]["text"])
```

---

## Lab 2: Self-hosted vLLM on EKS

```bash
# Deploy vLLM
kubectl apply -f vllm-deployment.yaml

# Test
curl http://vllm-service:8000/v1/completions \
  -d '{"model":"meta-llama/Llama-2-7b-chat-hf","prompt":"Hello","max_tokens":50}'
```

---

## Lab 3: LoRA Adapter Training + Deployment

```python
# Train adapter (see 04-FineTuning-Deployment.md)
# Deploy with vLLM --enable-lora
# Test switching between adapters via model name
```

---

## Lab 4: Blue/Green Model Update

```bash
# Deploy blue (v1) and green (v2)
# Switch service selector
# Verify zero-downtime switch
# Rollback test
```

---

## Lab 5: Cost Comparison Benchmark

```python
# Run same 1000 requests on:
# 1. Bedrock Claude Haiku
# 2. Bedrock Claude Sonnet
# 3. Self-hosted LLaMA-7B (vLLM)
# Compare: latency, quality, cost per request
```

---

## Summary Checklist
- [ ] Lab 1: Bedrock API working
- [ ] Lab 2: Self-hosted vLLM serving on EKS
- [ ] Lab 3: LoRA adapter deployed and switchable
- [ ] Lab 4: Blue/Green deployment tested with rollback
- [ ] Lab 5: Cost comparison documented
