# 10 — Cost Optimization & Multi-Tenancy

## GPU Costs (Reality Check)

```
Instance         │ GPUs      │ On-Demand/hr │ Spot/hr    │ Monthly (OD)
─────────────────┼───────────┼──────────────┼────────────┼─────────────
p5.48xlarge      │ 8× H100   │ $98.32       │ ~$40-60    │ $71,750
p4d.24xlarge     │ 8× A100   │ $32.77       │ ~$12-15    │ $23,922
g5.48xlarge      │ 8× A10G   │ $16.29       │ ~$6-8      │ $11,891
trn1.32xlarge    │ 16 Trainium│ $21.50      │ ~$8-10     │ $15,695

1 idle p5 node for 1 day = $2,360 WASTED
10 idle GPUs for a week = $16,500+ WASTED

→ Cost optimization is NOT optional at scale
```

## Cost Optimization Strategies

### 1. Spot Instances for Training

```yaml
# Karpenter NodePool with Spot
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-training-spot
spec:
  template:
    spec:
      requirements:
      - key: "karpenter.sh/capacity-type"
        operator: In
        values: ["spot"]                    # ← Spot instances!
      - key: "node.kubernetes.io/instance-type"
        operator: In
        values: ["p4d.24xlarge", "p4de.24xlarge"]  # Multiple types
      - key: "topology.kubernetes.io/zone"
        operator: In
        values: ["us-east-1a", "us-east-1b"]
  disruption:
    consolidationPolicy: WhenUnderutilized
    expireAfter: 24h

---
# Spot interruption handler (DaemonSet)
# When AWS sends 2-min warning:
# 1. Save checkpoint immediately
# 2. Gracefully terminate training
# 3. Karpenter launches new spot/on-demand node
# 4. Training resumes from checkpoint
```

### Spot + Checkpointing Strategy

```python
import signal
import sys

# Handle spot interruption (SIGTERM)
def spot_interruption_handler(signum, frame):
    print("SPOT INTERRUPTION! Saving emergency checkpoint...")
    save_checkpoint(model, optimizer, step, '/checkpoints/emergency.pt')
    sys.exit(0)

signal.signal(signal.SIGTERM, spot_interruption_handler)

# Training with frequent checkpoints for spot safety
CHECKPOINT_INTERVAL = 500  # Save every 500 steps (vs 2000 for on-demand)
# Trade-off: More frequent saves = more overhead but less lost work on interruption
```

### 2. Auto-Scaling (Scale to Zero)

```yaml
# Karpenter: Scale GPU nodes to zero when no jobs
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-inference
spec:
  template:
    spec:
      requirements:
      - key: "node.kubernetes.io/instance-type"
        operator: In
        values: ["g5.xlarge"]
  limits:
    nvidia.com/gpu: 32            # Max 32 GPUs
  disruption:
    consolidationPolicy: WhenEmpty # Remove nodes when no pods
    consolidateAfter: 5m           # Wait 5 min before removing

# Result:
# No inference requests → all GPU nodes terminate → $0
# Request comes in → Karpenter launches GPU node (~2 min)
# Trade-off: Cold start latency vs cost savings
```

### 3. Right-Sizing

```
Step 1: Profile your workload
├── Run on smallest GPU first (T4 or g5.xlarge)
├── Measure: GPU util, memory usage, throughput
├── If GPU util < 50% → model doesn't need bigger GPU
└── If OOM → move up to next tier

Step 2: Choose correct instance

Training:
├── Small model (<1B) → g5.xlarge (1× A10G, 24GB) = $1.00/hr
├── Medium model (1-7B) → p4d.24xlarge (8× A100, 80GB) = $32.77/hr
├── Large model (7-70B) → p5.48xlarge (8× H100, 80GB) = $98.32/hr
├── Cost-optimized → trn1.32xlarge (16× Trainium) = $21.50/hr
└── Don't use p5 if p4d works!

Inference:
├── Small model → g5.xlarge (1× A10G) = $1.00/hr
├── Medium model → g5.2xlarge (1× A10G, more CPU) = $1.21/hr
├── Large model → p4d.24xlarge with MIG = $32.77/hr (shared)
└── Cost-optimized → inf2.xlarge (Inferentia2) = $0.76/hr
```

### 4. Idle GPU Detection & Alerting

```yaml
# Prometheus alert: GPU idle for >30 min
- alert: GPUIdleWaste
  expr: |
    avg_over_time(DCGM_FI_DEV_GPU_UTIL{pod!=""}[30m]) < 5
    and on(pod) kube_pod_status_phase{phase="Running"}
  for: 5m
  labels:
    severity: warning
    team: "{{ $labels.namespace }}"
  annotations:
    summary: "Pod {{ $labels.pod }} GPU idle for 30+ min"
    cost_waste: "~$4/hr per A100"
    action: "Terminate or resize pod"
```

### 5. Time-Based Scheduling

```yaml
# Dev GPUs: Available only during work hours (9 AM - 9 PM)
# Night time: Scale to zero, save ~50% cost

apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: dev-gpu-daytime
spec:
  template:
    spec:
      requirements:
      - key: "workload-type"
        operator: In
        values: ["development"]
  limits:
    nvidia.com/gpu: 16           # Max 16 GPUs during day
  
# Combine with CronJob that cordons/uncordons or adjusts limits
# Or: Use Volcano queue that pauses dev queue at night
```

## Multi-Tenancy

### The Problem
```
3 teams, 80 GPUs total:
├── Team A (Research): Needs burst, can wait
├── Team B (Production): Needs guaranteed, critical
├── Team C (Dev): Small jobs, flexible
│
├── Without multi-tenancy: One team hogs all GPUs, others starve
└── With multi-tenancy: Fair sharing + guaranteed minimums
```

### Namespace-Based Isolation

```yaml
# Namespace per team
apiVersion: v1
kind: Namespace
metadata:
  name: team-research
  labels:
    team: research

---
apiVersion: v1
kind: Namespace
metadata:
  name: team-production
  labels:
    team: production

---
# ResourceQuota per namespace
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: team-research
spec:
  hard:
    requests.nvidia.com/gpu: "32"    # Max 32 GPUs
    limits.nvidia.com/gpu: "32"
    pods: "50"                        # Max 50 pods

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: team-production
spec:
  hard:
    requests.nvidia.com/gpu: "24"    # Max 24 GPUs (guaranteed)
    limits.nvidia.com/gpu: "24"
    pods: "100"
```

### Volcano Queues for Fair Sharing

```yaml
# Advanced: Volcano queues with borrowing
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: research
spec:
  weight: 4                    # Higher weight = more share
  guarantee:
    nvidia.com/gpu: 16         # Minimum guaranteed
  capability:
    nvidia.com/gpu: 48         # Can burst up to 48
  reclaimable: true            # Give back when others need

---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: production
spec:
  weight: 3
  guarantee:
    nvidia.com/gpu: 24         # Always guaranteed!
  capability:
    nvidia.com/gpu: 24         # Cannot burst beyond
  reclaimable: false           # NEVER give these away

---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: development
spec:
  weight: 1
  guarantee:
    nvidia.com/gpu: 0          # No guarantee
  capability:
    nvidia.com/gpu: 24         # Can use up to 24 if free
  reclaimable: true            # First to give back

# Behavior:
# Idle cluster: dev can use up to 24 GPUs (from free pool)
# Research submits job: dev GPUs get reclaimed → research gets them
# Production always has 24 (never touched)
```

### Cost Allocation & Chargeback

```yaml
# Label all pods with team/project for cost tracking
metadata:
  labels:
    team: research
    project: llm-training
    cost-center: "CC-1234"

# Kubecost or CloudHealth tracks:
# - GPU hours per team
# - GPU hours per project
# - Idle time per team
# - Cost per training job
```

### Chargeback Calculation

```
Monthly GPU cost: $50,000 (cluster)

Team allocation:
├── Research: Used 40% of GPU-hours → Charged $20,000
├── Production: Used 35% → Charged $17,500
├── Development: Used 15% → Charged $7,500
├── Idle waste: 10% → Shared across teams
└── Total: $50,000

Per-job cost:
  Job used 8 GPUs for 24 hours on p4d.24xlarge
  = 8 GPUs × 24 hrs × ($32.77/8 GPUs/hr) = $786
```

## Cost Dashboard Metrics

```
Key metrics to track:
├── GPU Utilization % (per team)
├── GPU Hours Consumed (per team/project)
├── Idle GPU Hours (waste)
├── Cost per Training Run
├── Spot vs On-Demand ratio
├── Queue Wait Time (are teams waiting too long?)
└── Cost per Step/Token (training efficiency)
```

## Interview Questions

1. **Q: 80 GPU cluster ka monthly cost $60K hai. 20% idle time hai. Kaise reduce karoge?**
   A: (1) Idle detection alerts + auto-terminate after 30 min idle, (2) Spot instances for training (40-60% savings), (3) Scale dev nodes to zero at night/weekends, (4) Right-size: check if teams actually need the GPU tier they're using, (5) Backfill small jobs into idle gaps. Target: <5% idle = save $9K/month.

2. **Q: Research team complain kar rahi hai ki dev team ne saare GPUs le liye. Fix?**
   A: Volcano queues with guarantees. Research queue: guarantee=16 GPUs, capability=48. Dev queue: guarantee=0, capability=24, reclaimable=true. When research submits jobs, dev GPUs automatically reclaimed and given to research.

3. **Q: Spot interruption aayi training ke beech. Data loss kaise minimize karo?**
   A: (1) Checkpoint every 500 steps (not 2000), (2) SIGTERM handler that saves emergency checkpoint, (3) Use on-demand for last 10% of training (when close to completion), (4) Multi-AZ spot diversification (different AZ = less likely all interrupted together).

4. **Q: Per-job cost tracking kaise implement karoge?**
   A: (1) Mandatory labels on all pods (team, project, cost-center), (2) GPU hours per pod tracked via Prometheus, (3) Kubecost for cost allocation, (4) Weekly chargeback report per team, (5) Budget alerts when team exceeds monthly allocation.
