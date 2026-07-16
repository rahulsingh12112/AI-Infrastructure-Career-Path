# 05 — GPU Scheduling: Volcano, Gang Scheduling & Advanced Scheduling

## The Problem with Default K8s Scheduler

```
Default K8s scheduler:
├── Schedules pods ONE at a time
├── No concept of "job" (group of pods)
├── No gang scheduling (all-or-nothing)
└── No fair-share queuing

Problem for ML:
  Distributed training needs 4 pods (each with 8 GPUs) = 32 GPUs
  Default scheduler: schedules pod 1... waits... schedules pod 2...
  What if only 24 GPUs free? 
  → 3 pods scheduled, 1 pending
  → 3 pods WASTING GPU doing nothing (can't train without all 4)!

  Solution: Gang scheduling — either ALL 4 pods get GPUs or NONE do.
```

## Volcano Scheduler

### What is Volcano?
Open-source batch system for Kubernetes, built for ML/HPC/Big Data.
Provides: gang scheduling, queue management, fair-share, priority preemption.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Volcano Components                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Scheduler   │  │  Controller  │  │   Admission   │  │
│  │               │  │              │  │   Webhook     │  │
│  │ - Gang sched  │  │ - Job CRD    │  │ - Validates   │  │
│  │ - Fair share  │  │ - Queue CRD  │  │   jobs/queues │  │
│  │ - Priority    │  │ - PodGroup   │  │               │  │
│  │ - Preemption  │  │   lifecycle  │  │               │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  CRDs:                                                   │
│  ├── vcjob.volcano.sh (VolcanoJob)                      │
│  ├── queue.volcano.sh (Queue)                           │
│  └── podgroup.volcano.sh (PodGroup)                     │
└─────────────────────────────────────────────────────────┘
```

### Installation

```bash
# Install Volcano
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml

# Verify
kubectl get pods -n volcano-system
# volcano-admission-xxxxx
# volcano-controllers-xxxxx
# volcano-scheduler-xxxxx
```

### Gang Scheduling Example

```yaml
# Distributed training: needs ALL 4 workers or NONE
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: distributed-training
spec:
  minAvailable: 4          # ← GANG: need all 4 pods!
  schedulerName: volcano
  queue: training-queue
  policies:
  - event: PodEvicted
    action: RestartJob
  - event: PodFailed
    action: RestartJob
  tasks:
  - replicas: 4
    name: worker
    template:
      spec:
        containers:
        - name: pytorch
          image: my-training:latest
          resources:
            limits:
              nvidia.com/gpu: 8
          env:
          - name: WORLD_SIZE
            value: "4"
        restartPolicy: OnFailure
```

### Queue Management

```yaml
# Create queues with resource quotas
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: training-queue
spec:
  weight: 3           # 3x priority over weight-1 queues
  reclaimable: true   # Can give resources to other queues when idle
  capability:
    nvidia.com/gpu: 64  # Max 64 GPUs for this queue

---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: inference-queue
spec:
  weight: 1
  reclaimable: true
  capability:
    nvidia.com/gpu: 32

---
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: dev-queue
spec:
  weight: 1
  reclaimable: true
  guarantee:
    nvidia.com/gpu: 8    # Minimum guaranteed GPUs
  capability:
    nvidia.com/gpu: 16   # Maximum GPUs
```

### Priority & Preemption

```yaml
# High priority training job can preempt low priority jobs
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: training-critical
value: 1000000
globalDefault: false
description: "Critical training jobs"

---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: dev-workload
value: 100
globalDefault: true
description: "Development workloads - can be preempted"

---
# Job using priority:
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: urgent-training
spec:
  minAvailable: 2
  schedulerName: volcano
  queue: training-queue
  priorityClassName: training-critical  # ← Will preempt dev jobs!
  tasks:
  - replicas: 2
    name: worker
    template:
      spec:
        priorityClassName: training-critical
        containers:
        - name: pytorch
          image: my-training:latest
          resources:
            limits:
              nvidia.com/gpu: 8
```

### Volcano Plugins (Scheduling Algorithms)

```yaml
# Volcano scheduler configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: volcano-scheduler-configmap
  namespace: volcano-system
data:
  volcano-scheduler.conf: |
    actions: "enqueue, allocate, preempt, backfill"
    tiers:
    - plugins:
      - name: gang              # Gang scheduling
        enablePreemptable: true
      - name: priority          # Priority-based ordering
      - name: conformance
    - plugins:
      - name: drf               # Dominant Resource Fairness
      - name: predicates        # Node filtering
      - name: proportion        # Queue proportion
      - name: nodeorder         # Node scoring
      - name: binpack           # Pack pods on fewer nodes (save cost)
```

## Apache YuniKorn (Alternative)

```
Volcano vs YuniKorn:
├── Volcano: ML/HPC focused, simpler, gang scheduling native
├── YuniKorn: More general, hierarchical queues, better for mixed workloads
└── Both support gang scheduling

YuniKorn is used by: Cloudera, Apple (internal), LinkedIn
Volcano is used by: Most ML platform teams
```

## Default K8s GPU Scheduling (Without Volcano)

### Resource Requests & Limits
```yaml
# Basic GPU request
resources:
  limits:
    nvidia.com/gpu: 2    # Pod gets 2 GPUs
    # Note: gpu is ONLY in limits, not requests
    # Kubernetes treats GPU as non-shareable
    # You get EXACTLY what you request
```

### Node Taints & Tolerations
```yaml
# Taint GPU nodes (prevent non-GPU pods from landing)
kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule

# Pod that tolerates the taint:
spec:
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Equal"
    value: "present"
    effect: "NoSchedule"
  containers:
  - name: training
    resources:
      limits:
        nvidia.com/gpu: 8
```

### Node Affinity for GPU Type
```yaml
# I want specifically A100 nodes, not T4
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: nvidia.com/gpu.product
            operator: In
            values: ["NVIDIA-A100-SXM4-80GB"]
          - key: nvidia.com/gpu.count
            operator: Gt
            values: ["7"]  # Only 8-GPU nodes
```

## Scheduling Decision Flowchart

```
New GPU Pod arrives
├── Is it part of a gang (minAvailable > 1)?
│   ├── YES → Check: Are ALL required resources available?
│   │   ├── YES → Schedule all pods together
│   │   └── NO → Wait in queue (don't schedule any)
│   └── NO → Normal scheduling
├── Which queue?
│   ├── Check queue capacity → is there room?
│   ├── Check queue weight → fair-share calculation
│   └── Check guarantee → minimum always available
├── Priority?
│   ├── Higher priority → schedule first
│   └── Can preempt lower priority? → evict if needed
├── Which node?
│   ├── Has enough GPUs free?
│   ├── Topology: NVLink-connected GPUs available?
│   ├── NUMA alignment possible?
│   └── Placement group constraint met?
└── Schedule!
```

## Real-World Multi-Tenancy Setup

```yaml
# Company with 3 teams sharing GPU cluster:
#
# Team A (Research): 40% cluster, can burst to 60%
# Team B (Production Inference): 30% cluster, guaranteed 30%
# Team C (Development): 30% cluster, can be preempted

# Queue config:
Queues:
  research-queue:
    weight: 4
    guarantee: {gpu: 32}   # Always get 32 GPUs
    capability: {gpu: 64}  # Can burst to 64
    reclaimable: true

  production-queue:
    weight: 3
    guarantee: {gpu: 24}   # Always get 24
    capability: {gpu: 32}
    reclaimable: false      # Never give away!

  dev-queue:
    weight: 3
    guarantee: {gpu: 0}    # No guarantee
    capability: {gpu: 24}
    reclaimable: true       # Will give up when others need

# Total cluster: 80 GPUs
# Idle time: dev can use up to 24 (from reclaimable pools)
# Busy time: production guaranteed 24, research guaranteed 32
```

## Interview Questions

1. **Q: Gang scheduling nahi use kiya aur distributed training mein 1 pod pending reh gaya. Kya hoga?**
   A: Baaki 3 pods GPU reserve karke idle baithenge (NCCL initialization mein wait karenge, eventually timeout). GPUs waste + training doesn't start. Gang scheduling prevents this by ensuring all-or-nothing.

2. **Q: 80 GPU cluster pe 3 teams hain. Fair sharing kaise implement karoge?**
   A: Volcano queues with DRF (Dominant Resource Fairness) plugin. Each team gets a queue with weight + guarantee + capability limits. Reclaimable flag ensures idle GPUs are shared but guaranteed resources are always available.

3. **Q: Production inference job ko training job ne preempt kar diya. Kaise rokoge?**
   A: Production queue ko `reclaimable: false` set karo + higher PriorityClass do. Ya better: separate node pools with taints — training nodes alag, inference nodes alag, physically isolated.

4. **Q: Backfill scheduling kya hai?**
   A: Queue mein 32-GPU job wait kar raha hai (not enough GPUs). Meanwhile 4-GPU job bhi wait kar raha hai jo fit ho sakta hai. Backfill = chhote job ko pehle schedule karo agar wo complete ho jayega before bade job ke resources free hone tak. GPUs idle nahi rehte.
