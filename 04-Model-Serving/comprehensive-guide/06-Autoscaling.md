# 06 — Autoscaling Inference Endpoints

## Why Autoscaling for LLM Inference?

```
LLM traffic patterns:
├── Peak hours: 10x normal traffic (work hours)
├── Off-peak: Near zero (nighttime)
├── Burst: Viral content → sudden 50x spike
├── GPU cost: $1-100/hr per instance
│
├── Without autoscaling: Pay for peak 24/7 = $$$$ waste
├── With autoscaling: Scale up for peak, down for off-peak = $$ savings
└── Key challenge: GPU cold start is SLOW (2-10 min to load LLM)
```

## Autoscaling Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              LLM Inference Autoscaling                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Load Balancer                                               │
│  ├── Route requests to healthy instances                    │
│  ├── Health checks (model loaded + responding)              │
│  └── Connection draining (graceful scale-down)              │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                    │
│  │Instance 1│ │Instance 2│ │Instance 3│ (scale up/down)     │
│  │  vLLM    │ │  vLLM    │ │  vLLM    │                    │
│  │  GPU: A10│ │  GPU: A10│ │  GPU: A10│                    │
│  └──────────┘ └──────────┘ └──────────┘                    │
│                                                              │
│  Autoscaler (watches metrics):                              │
│  ├── Scale UP when: queue > threshold OR latency > SLA      │
│  ├── Scale DOWN when: GPU util < 20% for 10 min            │
│  ├── Min replicas: 1 (always on)                            │
│  └── Max replicas: 10 (budget cap)                          │
│                                                              │
│  Metrics (what to scale on):                                │
│  ├── Request queue depth (best for LLMs)                    │
│  ├── GPU utilization (traditional but misleading for LLMs)  │
│  ├── Tokens per second (throughput-based)                   │
│  ├── P99 latency (SLA-based)                                │
│  └── Concurrent requests (simplest)                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Scaling Metrics for LLMs

```
Metric                  │ Good for LLMs? │ Why
────────────────────────┼────────────────┼──────────────────────────────
GPU Utilization         │ ⚠️ Misleading  │ LLM decode is memory-bound, GPU util
                        │                │ shows low even when fully saturated
Queue Depth             │ ✅ Best        │ Directly measures demand backlog
Concurrent Requests     │ ✅ Good        │ Simple, predictable
P99 Latency             │ ✅ Good        │ SLA-driven, reactive
Tokens/Second           │ ✅ Good        │ Measures actual throughput
KV-Cache Usage %        │ ✅ Best (LLM)  │ When KV-cache full → can't serve more
CPU Utilization         │ ❌ Bad         │ LLM is GPU-bound, CPU always low
Request Rate (RPS)      │ ⚠️ Okay       │ Doesn't account for request complexity
```

## Kubernetes HPA for LLM

### Custom Metrics Based Autoscaling

```yaml
# HPA based on queue depth (best for LLMs)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 1
  maxReplicas: 10
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60    # React quickly to demand
      policies:
      - type: Pods
        value: 2                         # Add max 2 pods at a time
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300   # Wait 5 min before scale down
      policies:
      - type: Pods
        value: 1                         # Remove 1 pod at a time
        periodSeconds: 120
  metrics:
  # Scale on queue depth
  - type: Pods
    pods:
      metric:
        name: tgi_queue_size             # Or vllm_queue_size
      target:
        type: AverageValue
        averageValue: "5"                # Scale up if avg queue > 5

  # Also consider latency
  - type: Pods
    pods:
      metric:
        name: request_latency_p99_seconds
      target:
        type: AverageValue
        averageValue: "2"                # Scale up if p99 > 2 sec
```

### Karpenter for GPU Node Scaling

```yaml
# When HPA adds pods, Karpenter provisions GPU nodes
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: inference-gpu
spec:
  template:
    spec:
      requirements:
      - key: "node.kubernetes.io/instance-type"
        operator: In
        values: ["g5.xlarge", "g5.2xlarge"]
      - key: "karpenter.sh/capacity-type"
        operator: In
        values: ["on-demand"]    # On-demand for inference (no interruption)
  limits:
    nvidia.com/gpu: 10            # Max 10 GPU nodes
  disruption:
    consolidationPolicy: WhenUnderutilized
    consolidateAfter: 10m
```

## SageMaker Autoscaling

```python
import boto3

aas = boto3.client("application-autoscaling")

# Register endpoint as scalable target
aas.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/llm-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=10,
)

# Target tracking: Scale based on concurrent requests per instance
aas.put_scaling_policy(
    PolicyName="concurrent-requests-scaling",
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/llm-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 10.0,    # Target: 10 concurrent requests per instance
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantConcurrentRequestsPerModelHighResolution",
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60,
    },
)
```

## Cold Start Mitigation

```
Problem: New LLM instance takes 3-10 minutes to start:
├── Node provisioning: 1-2 min
├── Container pull: 1-3 min (large images, 20-50GB)
├── Model loading to GPU: 1-5 min
└── Total: 3-10 min cold start!

Solutions:

1. Minimum replicas (always warm):
   minReplicas: 2  # Never scale below 2

2. Pre-pulled images:
   # Karpenter userData: pre-pull container on node start
   # Or: Use EBS snapshots with model pre-loaded

3. Model caching (shared storage):
   # Mount model from FSx/EFS (no download from HuggingFace)
   # Local NVMe cache for fastest loading

4. Predictive scaling:
   # Scale up BEFORE peak hours (scheduled)
   # AWS: ScheduledAction at 8 AM, scale down at 8 PM

5. Warm pool (spare instances):
   # Keep 1-2 instances with model loaded but no traffic
   # Instantly absorb spikes

6. Smaller models for buffer:
   # During scale-up: Route to smaller/quantized model
   # Once big instance ready: Route back
```

### Scheduled Scaling
```python
# Scale up before peak hours (predictable traffic)
aas.put_scheduled_action(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/llm-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    ScheduledActionName="scale-up-morning",
    Schedule="cron(0 8 * * ? *)",     # 8 AM daily
    ScalableTargetAction={
        "MinCapacity": 5,
        "MaxCapacity": 10,
    },
)

aas.put_scheduled_action(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/llm-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    ScheduledActionName="scale-down-night",
    Schedule="cron(0 22 * * ? *)",    # 10 PM daily
    ScalableTargetAction={
        "MinCapacity": 1,
        "MaxCapacity": 3,
    },
)
```

## Load Balancing for LLMs

```
Challenge: LLM requests are NOT equal!
├── Short request (10 token output): 200ms
├── Long request (2000 token output): 30 seconds
├── Round-robin load balancing: One instance gets all long requests → overloaded!

Solutions:
├── Least-connections: Route to instance with fewest active requests
├── Weighted by KV-cache usage: Route to instance with most free memory
├── Request-aware: Estimate output length, balance accordingly
└── vLLM handles this internally (continuous batching manages it)
```

## Interview Questions

1. **Q: LLM inference endpoint autoscale karna hai. Konsa metric use karoge?**
   A: Primary: Queue depth (requests waiting). Secondary: P99 latency. NOT GPU utilization (misleading for LLMs — decode phase is memory-bound, shows low GPU util even when saturated). Queue depth directly measures unmet demand. Scale up when avg queue > 5, scale down when queue = 0 for 5+ minutes.

2. **Q: Cold start 5 min hai. Scale-up event pe users ko 5 min wait karna padega?**
   A: Mitigation: (1) Keep minReplicas=2 (always warm), (2) Predictive/scheduled scaling (scale before peak), (3) Pre-pulled container images + model on local NVMe/FSx, (4) During cold start: queue requests (don't reject), existing instances handle increased load temporarily, (5) Consider warm pool (spare loaded instance).

3. **Q: Scale-down kab karna safe hai? Active requests ka kya hoga?**
   A: (1) Connection draining: Stop routing NEW requests to instance, (2) Wait for active requests to complete (terminationGracePeriodSeconds: 300), (3) Then terminate. Scale-down cooldown: 5-10 min (avoid flapping). Never scale below minReplicas. In K8s: Use PDB (PodDisruptionBudget) to prevent killing pods with active generations.

4. **Q: Traffic 10x spike aaya suddenly. Autoscaler fast enough nahi hai. Kya karoge?**
   A: (1) Rate limiting + queueing (don't crash, just queue), (2) Shed load gracefully (return 429 with retry-after), (3) Reduce max_tokens for all requests during spike, (4) If predictable (marketing launch): pre-scale, (5) Serverless overflow (route excess to SageMaker serverless as backup), (6) Multi-region: route traffic to less loaded region.
