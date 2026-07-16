# 02 — MIG vs MPS — GPU Partitioning

## The Problem

A100 GPU = 80GB memory, 6912 CUDA cores
Inference job = uses only 5GB memory, 10% compute

→ 90% GPU wasted! How to share 1 GPU among multiple workloads?

## Two Solutions

```
┌──────────────────────────────────────────────────────┐
│                    1 x A100 GPU                       │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Option 1: MIG (Hardware Partitioning)                │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│  │ 1g  │ │ 1g  │ │ 1g  │ │ 1g  │ │ 1g  │ │ 1g  │ │ 1g  │
│  │10GB │ │10GB │ │10GB │ │10GB │ │10GB │ │10GB │ │10GB │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘
│  7 isolated instances — each with own memory & compute │
│                                                       │
│  Option 2: MPS (Software Sharing)                     │
│  ┌───────────────────────────────────────────────┐    │
│  │         Shared GPU (no isolation)              │    │
│  │  Process A ──┐                                 │    │
│  │  Process B ──┼── All share same GPU memory     │    │
│  │  Process C ──┘                                 │    │
│  └───────────────────────────────────────────────┘    │
│                                                       │
└──────────────────────────────────────────────────────┘
```

## MIG (Multi-Instance GPU) — Deep Dive

### What is MIG?
Hardware-level GPU partitioning. Available on A100, A30, H100 only.
Each partition (called "GPU Instance") has:
- Dedicated compute (SMs)
- Dedicated memory
- Dedicated memory bandwidth
- Separate error isolation

### MIG Profiles (A100 80GB)

```
Profile    │ SMs │ Memory │ Use Case
───────────┼─────┼────────┼──────────────────────
7g.80gb    │ All │ 80 GB  │ Full GPU (no partition)
4g.40gb    │ 4/7 │ 40 GB  │ Large inference model
3g.40gb    │ 3/7 │ 40 GB  │ Medium training
2g.20gb    │ 2/7 │ 20 GB  │ Medium inference
1g.10gb    │ 1/7 │ 10 GB  │ Small inference model
1g.10gb+me │ 1/7 │ 10 GB  │ Same + media extensions
```

### Valid Partition Combinations (A100)

```
1 x 7g.80gb                     (full GPU, no MIG)
1 x 4g.40gb + 1 x 3g.40gb      (2 partitions)
1 x 4g.40gb + 2 x 1g.10gb      (3 partitions)  — NOT VALID!
3 x 2g.20gb + 1 x 1g.10gb      (4 partitions)
7 x 1g.10gb                     (7 partitions, max)

Note: Not all combinations are valid! Memory/SM slots have constraints.
```

### MIG Commands

```bash
# Enable MIG mode (requires GPU reset)
sudo nvidia-smi -i 0 -mig 1

# Check MIG status
nvidia-smi mig -lgip   # List GPU Instance Profiles
nvidia-smi mig -lgi    # List created GPU Instances

# Create GPU Instances
nvidia-smi mig -cgi 19,19,19,19,19,19,19  # 7 x 1g.10gb
nvidia-smi mig -cgi 9,14                    # 1 x 4g.40gb + 1 x 3g.40gb

# Create Compute Instances (within GPU Instance)
nvidia-smi mig -cci    # Create compute instance for each GI

# Destroy
nvidia-smi mig -dci    # Destroy compute instances
nvidia-smi mig -dgi    # Destroy GPU instances

# Disable MIG
sudo nvidia-smi -i 0 -mig 0
```

### MIG on Kubernetes

```yaml
# GPU Operator MIG config (ConfigMap)
apiVersion: v1
kind: ConfigMap
metadata:
  name: mig-parted-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    mig-configs:
      all-1g.10gb:
        - device-filter: ["0x20B210DE"]  # A100
          devices: all
          mig-enabled: true
          mig-devices:
            "1g.10gb": 7
      all-balanced:
        - device-filter: ["0x20B210DE"]
          devices: all
          mig-enabled: true
          mig-devices:
            "3g.40gb": 1
            "2g.20gb": 1
            "1g.10gb": 2

---
# Pod requesting specific MIG slice:
apiVersion: v1
kind: Pod
metadata:
  name: mig-pod
spec:
  containers:
  - name: cuda-app
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    resources:
      limits:
        nvidia.com/mig-1g.10gb: 1   # Request 1 MIG slice
```

### MIG Strategies on K8s

```
Strategy     │ Description                          │ Use Case
─────────────┼──────────────────────────────────────┼────────────────
single       │ All GPUs same MIG profile            │ Uniform workloads
mixed        │ Different profiles per GPU           │ Mixed workloads
none         │ MIG disabled, whole GPUs             │ Training only
```

## MPS (Multi-Process Service) — Deep Dive

### What is MPS?
Software-level GPU sharing. Available on ALL NVIDIA GPUs.
Multiple processes share the same GPU simultaneously via a server process.

```
┌─────────────────────────────────────────┐
│              MPS Architecture             │
├─────────────────────────────────────────┤
│                                          │
│  Client Process A ──┐                    │
│  Client Process B ──┼── MPS Server ── GPU│
│  Client Process C ──┘     (daemon)       │
│                                          │
│  MPS Server:                             │
│  - Multiplexes CUDA contexts             │
│  - Shares GPU compute across clients     │
│  - Single CUDA context (less overhead)   │
└─────────────────────────────────────────┘
```

### MPS on Kubernetes (Time-Slicing)

```yaml
# Device plugin config for GPU sharing (time-slicing)
apiVersion: v1
kind: ConfigMap
metadata:
  name: device-plugin-config
  namespace: gpu-operator
data:
  config.yaml: |
    version: v1
    sharing:
      timeSlicing:
        renameByDefault: false
        resources:
        - name: nvidia.com/gpu
          replicas: 4    # Each GPU appears as 4 "virtual" GPUs
```

## MIG vs MPS — Detailed Comparison

```
Feature          │ MIG                        │ MPS/Time-Slicing
─────────────────┼────────────────────────────┼──────────────────────
Isolation        │ Full (hardware)            │ None (software)
Memory isolation │ Yes (dedicated per slice)  │ No (shared, can OOM others)
Error isolation  │ Yes (fault in 1 ≠ affects  │ No (1 crash = all crash)
                 │ others)                    │
GPU support      │ A100, A30, H100 only       │ All NVIDIA GPUs
Max partitions   │ 7 (A100)                   │ Unlimited (but perf drops)
Overhead         │ Minimal                    │ Context switching cost
Flexibility      │ Fixed profiles only        │ Any ratio possible
Reconfiguration  │ Needs GPU reset            │ Dynamic, no restart
Best for         │ Production multi-tenant    │ Dev/test, burst sharing
```

## Decision Matrix — When to Use What

```
Scenario                              │ Use
──────────────────────────────────────┼─────────────────
Production inference, multiple models │ MIG
Multiple users, need isolation        │ MIG
Dev/test, not critical                │ MPS/Time-Slicing
GPU is underutilized, want to share   │ MPS/Time-Slicing
Large training job (needs full GPU)   │ Neither — whole GPU
H100/A100 available                   │ MIG preferred
Only T4/V100 available                │ MPS only option
Fault tolerance required              │ MIG (error isolation)
```

## Real-World Production Setup

```yaml
# Cluster with mixed strategy:
#
# Node pool 1: Training nodes (no partitioning)
#   - 8x A100, full GPU per pod
#   - Taint: workload=training:NoSchedule
#
# Node pool 2: Inference nodes (MIG)
#   - 8x A100, each split 7x 1g.10gb = 56 virtual GPUs
#   - Taint: workload=inference:NoSchedule
#
# Node pool 3: Dev nodes (Time-slicing)
#   - 4x T4, each shared 4-way = 16 virtual GPUs
#   - No taint (default for dev workloads)
```

## Interview Questions

1. **Q: Ek A100 pe 10 small inference models run karne hain. MIG ya MPS?**
   A: MIG — kyunki max 7 slices ban sakte hain, 7 MIG instances banao. Baaki 3 models ke liye ya toh 2g.20gb slices use karo (fit 2 models per slice) ya second GPU lagao. MIG isliye kyunki production mein isolation zaroori hai.

2. **Q: MIG enabled karne ke liye GPU reset kyun chahiye?**
   A: MIG hardware-level partitioning hai — memory controllers aur SM groups physically reconfigure hote hain. Ye running processes ke saath nahi ho sakta.

3. **Q: Time-slicing mein 1 pod crash hua toh baaki pe kya asar?**
   A: Poore GPU ka CUDA context corrupt ho sakta hai → saare sharing pods crash ho sakte hain. Isliye production mein MIG preferred hai.

4. **Q: H100 pe MIG vs A100 pe MIG — kya fark hai?**
   A: H100 pe confidential computing support hai MIG mein (encrypted memory per partition), aur more flexible profiles available hain.
