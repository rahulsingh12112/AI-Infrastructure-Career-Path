# 05 — Blue/Green & Canary Deployments for Models

## Why Blue/Green for LLMs?

```
Model update risks:
├── New model might be worse (quality regression)
├── Latency might increase (bigger model, bad config)
├── Memory leak or crash under load
└── Need instant rollback capability

Blue/Green: Run old + new simultaneously, switch traffic instantly
Canary: Route small % to new, monitor, gradually increase
```

## Blue/Green on Kubernetes

```yaml
# Blue (current production) — v1
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-blue
  labels:
    version: blue
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
      version: blue
  template:
    metadata:
      labels:
        app: vllm
        version: blue
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args: ["--model", "/models/llama-v1"]
        resources:
          limits:
            nvidia.com/gpu: 1

---
# Green (new version) — v2
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-green
  labels:
    version: green
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm
      version: green
  template:
    metadata:
      labels:
        app: vllm
        version: green
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args: ["--model", "/models/llama-v2"]
        resources:
          limits:
            nvidia.com/gpu: 1

---
# Service points to BLUE (switch to green by changing selector)
apiVersion: v1
kind: Service
metadata:
  name: vllm-prod
spec:
  selector:
    app: vllm
    version: blue    # ← Change to "green" to switch!
  ports:
  - port: 8000
```

## Canary with Istio

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vllm-canary
spec:
  hosts: [vllm-prod]
  http:
  - route:
    - destination:
        host: vllm-blue
      weight: 90          # 90% to old model
    - destination:
        host: vllm-green
      weight: 10          # 10% to new model (canary)
```

## Rollback Strategy

```
Monitor for 24-48 hours:
├── Quality metrics (human eval scores, automated eval)
├── Latency (TTFT, TPOT — should not regress)
├── Error rate (timeouts, OOM)
├── User feedback (thumbs up/down)

If regression detected:
├── Istio: Set green weight to 0% (instant)
├── K8s service: Switch selector back to blue (instant)
└── Total rollback time: < 30 seconds
```

## Interview Questions

1. **Q: Model update kaise karoge zero-downtime?**
   A: Blue/Green: Deploy new model as "green" alongside "blue". Validate green with synthetic traffic. Switch service selector. Keep blue running 24h for instant rollback.

2. **Q: Canary mein quality kaise measure karoge automatically?**
   A: (1) Shadow eval: Send same requests to both, compare outputs with LLM-as-judge, (2) A/B metrics: Track user engagement (thumbs up/down ratio), (3) Automated benchmarks: Run eval suite every hour on canary.
