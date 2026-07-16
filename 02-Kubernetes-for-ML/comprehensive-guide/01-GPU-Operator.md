# 01 — NVIDIA GPU Operator & Device Plugin

## What is GPU Operator?

GPU Operator automates the management of all NVIDIA software components needed to provision GPUs on Kubernetes:
- GPU drivers
- Container runtime (nvidia-container-toolkit)
- Device plugin
- DCGM exporter
- MIG manager
- GPU Feature Discovery

## Without GPU Operator vs With GPU Operator

```
WITHOUT (manual):
├── Install NVIDIA driver on every node manually
├── Install nvidia-container-toolkit manually
├── Install device plugin DaemonSet manually
├── Update drivers = node by node SSH
└── New node added? Repeat everything 😭

WITH GPU Operator:
├── Install operator via Helm (1 command)
├── Operator installs EVERYTHING automatically
├── New node added? Operator configures it automatically
├── Driver update? Change version in CR, operator handles rolling update
└── You just manage a single CustomResource ✅
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Kubernetes Cluster                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  GPU Operator (runs as deployment)                   │
│  ├── Watches: ClusterPolicy CR                       │
│  ├── Manages DaemonSets:                             │
│  │   ├── nvidia-driver-daemonset                     │
│  │   ├── nvidia-container-toolkit-daemonset          │
│  │   ├── nvidia-device-plugin-daemonset              │
│  │   ├── nvidia-dcgm-exporter                        │
│  │   ├── nvidia-mig-manager                          │
│  │   └── gpu-feature-discovery                       │
│  └── Validates: GPU node health                      │
│                                                      │
│  GPU Node:                                           │
│  ┌─────────────────────────────────────────────┐     │
│  │  Pod (requests nvidia.com/gpu: 1)           │     │
│  │  ├── Container sees GPU via device plugin    │     │
│  │  ├── nvidia-smi works inside container       │     │
│  │  └── CUDA libraries available                │     │
│  └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

## Key Components Deep Dive

### 1. NVIDIA Device Plugin
```yaml
# What it does:
# - Advertises GPUs to kubelet (nvidia.com/gpu resource)
# - Allocates specific GPU to container at runtime
# - Handles GPU health checks

# Pod requesting GPU:
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  containers:
  - name: cuda-container
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 1  # Request 1 GPU
```

### 2. GPU Feature Discovery (GFD)
```yaml
# Automatically labels nodes with GPU properties:
# nvidia.com/gpu.product = A100-SXM4-80GB
# nvidia.com/gpu.memory = 81920
# nvidia.com/gpu.count = 8
# nvidia.com/mig.capable = true
# nvidia.com/cuda.driver.major = 535

# Use these labels for node affinity:
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: nvidia.com/gpu.product
          operator: In
          values: ["A100-SXM4-80GB"]
```

### 3. DCGM Exporter
```
# Exports GPU metrics to Prometheus
# Key metrics:
DCGM_FI_DEV_GPU_UTIL        → GPU utilization %
DCGM_FI_DEV_FB_USED         → Framebuffer memory used (MB)
DCGM_FI_DEV_FB_FREE         → Framebuffer memory free (MB)
DCGM_FI_DEV_GPU_TEMP        → GPU temperature
DCGM_FI_DEV_POWER_USAGE     → Power draw (Watts)
DCGM_FI_DEV_SM_CLOCK        → SM clock speed
DCGM_FI_DEV_XID_ERRORS      → Hardware errors
DCGM_FI_PROF_SM_ACTIVE      → SM activity %
DCGM_FI_PROF_SM_OCCUPANCY   → SM occupancy %
```

## Installation

```bash
# 1. Add Helm repo
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# 2. Install GPU Operator
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator

# 3. Verify
kubectl get pods -n gpu-operator
kubectl get nodes -o json | jq '.items[].status.capacity'
# Should show: "nvidia.com/gpu": "8"
```

## ClusterPolicy CR (Main Configuration)

```yaml
apiVersion: nvidia.com/v1
kind: ClusterPolicy
metadata:
  name: cluster-policy
spec:
  operator:
    defaultRuntime: containerd
  driver:
    enabled: true
    version: "535.129.03"
  toolkit:
    enabled: true
  devicePlugin:
    enabled: true
    config:
      name: device-plugin-config
  dcgmExporter:
    enabled: true
  migManager:
    enabled: true
  gfd:
    enabled: true
```

## Upgrade Strategy

```bash
# Rolling update — one node at a time
# 1. Cordon node (no new pods)
# 2. Drain GPU pods
# 3. Update driver
# 4. Uncordon node
# Operator handles this automatically when you change driver version

# To update:
kubectl edit clusterpolicy cluster-policy
# Change spec.driver.version → new version
# Operator does rolling update automatically
```

## Real-World Scenarios

### Scenario 1: New GPU node added to cluster
```
1. Node joins cluster
2. GPU Operator detects: "new node has GPU"
3. DaemonSets roll out to new node:
   - Driver installed
   - Toolkit configured
   - Device plugin starts
   - GFD labels node
4. Node ready to accept GPU pods (5-10 minutes)
```

### Scenario 2: Driver crash on a node
```
1. DCGM detects GPU not responding
2. Node marked as unhealthy
3. GPU pods evicted
4. Operator attempts driver reload
5. If fails → node cordoned, alert fired
```

## Interview Questions

1. **Q: GPU Operator ke bina kya hoga?**
   A: Har node pe manually driver install karna padega, updates painful honge, new node add karna slow hoga. Production mein unmanageable at scale.

2. **Q: Device Plugin kaise decide karta hai konsa GPU kis pod ko milega?**
   A: Kubelet device plugin ko call karta hai → plugin available GPU list se allocate karta hai → container ko specific GPU device file (/dev/nvidia0) mount karta hai.

3. **Q: Agar 8 GPU node pe 4 pods chahiye, each with 2 GPUs?**
   A: Device plugin allocates 2 GPUs per pod. GFD + topology awareness ensure ki same-NVLink-connected GPUs ek pod ko milein for best performance.

4. **Q: Driver update kaise karte ho without downtime?**
   A: ClusterPolicy mein version change karo → Operator rolling update karta hai (1 node at a time, drain → update → uncordon).
