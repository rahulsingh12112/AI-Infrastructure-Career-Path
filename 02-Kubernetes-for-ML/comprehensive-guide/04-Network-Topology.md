# 04 — Network Topology & GPU-Aware Scheduling

## Why Topology Matters

Not all GPU connections are equal:
```
GPU0 ←NVLink→ GPU1 = 600 GB/s   (FAST)
GPU0 ←PCIe→ GPU4   = 64 GB/s    (10x SLOWER)
```

If scheduler puts communicating pods on wrong GPUs → training speed drops 10x.

## GPU Topology Inside a Node (DGX A100)

```
┌─────────────────────────────────────────────────────────┐
│                 DGX A100 (8 GPUs)                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   NVSwitch Fabric (ALL-to-ALL NVLink)                    │
│   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                      │
│   │NVSw0│ │NVSw1│ │NVSw2│ │NVSw3│ │NVSw4│ │NVSw5│     │
│   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                      │
│      │       │       │       │                           │
│   ┌──┴──┐ ┌──┴──┐ ┌──┴──┐ ┌──┴──┐                      │
│   │GPU 0│ │GPU 1│ │GPU 2│ │GPU 3│                       │
│   │GPU 4│ │GPU 5│ │GPU 6│ │GPU 7│                       │
│   └─────┘ └─────┘ └─────┘ └─────┘                      │
│                                                          │
│   DGX A100: Every GPU connected to every other           │
│   via NVSwitch = 600 GB/s any-to-any                     │
│                                                          │
│   Older systems (V100 DGX-1):                           │
│   Not all GPUs directly connected                        │
│   GPU0↔GPU1 = NVLink, GPU0↔GPU5 = PCIe (much slower)   │
│                                                          │
├─────────────────────────────────────────────────────────┤
│   PCIe Hierarchy:                                        │
│   ┌── CPU 0 (NUMA 0) ────────────────┐                  │
│   │   PCIe Switch 0: GPU0, GPU1       │                  │
│   │   PCIe Switch 1: GPU2, GPU3       │                  │
│   │   NIC 0 (InfiniBand/EFA)          │                  │
│   └───────────────────────────────────┘                  │
│   ┌── CPU 1 (NUMA 1) ────────────────┐                  │
│   │   PCIe Switch 2: GPU4, GPU5       │                  │
│   │   PCIe Switch 3: GPU6, GPU7       │                  │
│   │   NIC 1 (InfiniBand/EFA)          │                  │
│   └───────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

## nvidia-smi topo -m (Topology Matrix)

```bash
$ nvidia-smi topo -m

        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  NIC0  NIC1
GPU0     X    NV12  NV12  NV12  NV12  NV12  NV12  NV12  PXB   SYS
GPU1    NV12   X    NV12  NV12  NV12  NV12  NV12  NV12  PXB   SYS
GPU2    NV12  NV12   X    NV12  NV12  NV12  NV12  NV12  NODE  SYS
GPU3    NV12  NV12  NV12   X    NV12  NV12  NV12  NV12  NODE  SYS
GPU4    NV12  NV12  NV12  NV12   X    NV12  NV12  NV12  SYS   PXB
GPU5    NV12  NV12  NV12  NV12  NV12   X    NV12  NV12  SYS   PXB
GPU6    NV12  NV12  NV12  NV12  NV12  NV12   X    NV12  SYS   NODE
GPU7    NV12  NV12  NV12  NV12  NV12  NV12  NV12   X    SYS   NODE

Legend:
  NV12 = NVLink (12 links, full speed)
  PXB  = Same PCIe switch (fast PCIe)
  NODE = Same NUMA node (crosses PCIe switch)
  SYS  = Cross NUMA (crosses CPU socket, slowest)
```

### Why This Matters for Networking:
```
GPU0 → NIC0 = PXB (same PCIe switch) → GPUDirect RDMA works optimally
GPU4 → NIC0 = SYS (cross NUMA) → Data crosses CPU socket → slower

Best practice: Route GPU traffic through NEAREST NIC
GPU0-GPU3 → NIC0 (same NUMA)
GPU4-GPU7 → NIC1 (same NUMA)
```

## NUMA Awareness

```
NUMA (Non-Uniform Memory Access):
├── CPU has multiple sockets
├── Each socket has local memory (fast) and remote memory (slow)
├── GPUs + NICs attached to specific socket
└── Cross-socket = ~30% latency penalty

For AI workloads:
├── Pin GPU pod to correct NUMA node
├── Ensure NIC on same NUMA as GPUs
├── Avoid cross-socket memory access
└── Use numactl or K8s topology manager
```

### Kubernetes Topology Manager

```yaml
# kubelet config for topology-aware resource allocation
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
topologyManagerPolicy: "best-effort"    # or "restricted", "single-numa-node"
topologyManagerScope: "pod"             # or "container"

# Policies:
# none           → no topology alignment (default)
# best-effort    → try to align, but don't reject if can't
# restricted     → reject pod if can't align all resources
# single-numa-node → all resources MUST be on same NUMA node
```

## Topology-Aware Scheduling on K8s

### Method 1: Node Labels + Affinity
```yaml
# Nodes already labeled by GPU Feature Discovery:
# nvidia.com/gpu.product=A100-SXM4-80GB
# nvidia.com/gpu.count=8
# topology.kubernetes.io/zone=us-east-1a

apiVersion: v1
kind: Pod
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: nvidia.com/gpu.product
            operator: In
            values: ["A100-SXM4-80GB"]
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            job-name: "my-training"
        topologyKey: "kubernetes.io/hostname"  # Same node preferred
```

### Method 2: Topology Spread Constraints
```yaml
# For distributed training — spread across nodes but keep all GPUs
# within a node co-located
spec:
  topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: "kubernetes.io/hostname"
    whenUnsatisfiable: DoNotSchedule
    labelSelector:
      matchLabels:
        job: "distributed-training"
```

### Method 3: GPU Topology Device Plugin Config
```yaml
# NVIDIA device plugin with topology awareness
apiVersion: v1
kind: ConfigMap
metadata:
  name: device-plugin-config
data:
  config.yaml: |
    version: v1
    flags:
      migStrategy: none
      deviceListStrategy: envvar
      passDeviceSpecs: true
      gdsEnabled: false
      mofedEnabled: true
    sharing:
      timeSlicing:
        resources:
        - name: nvidia.com/gpu
          replicas: 1
    # Topology-aware allocation:
    # Device plugin will prefer allocating GPUs 
    # that are on same NVLink group
```

## Inter-Node Topology (Multi-Node Training)

```
For 4-node distributed training:

BEST topology (fat-tree network):
┌──────┐     ┌──────────┐     ┌──────┐
│Node 0│─────│  Spine   │─────│Node 2│
│8 GPUs│     │  Switch  │     │8 GPUs│
└──────┘     └──────────┘     └──────┘
┌──────┐          │           ┌──────┐
│Node 1│──────────┘───────────│Node 3│
│8 GPUs│                      │8 GPUs│
└──────┘                      └──────┘

All nodes same hop count → uniform bandwidth
= Best for AllReduce (everyone equal)

WORST topology (over-subscribed tree):
              ┌────────┐
              │ Core   │ ← bottleneck!
              └───┬────┘
         ┌────────┴────────┐
     ┌───┴───┐         ┌───┴───┐
     │ Leaf0 │         │ Leaf1 │
     └┬────┬─┘         └─┬───┬─┘
   Node0  Node1       Node2  Node3

Node0 ↔ Node1 = fast (same leaf)
Node0 ↔ Node2 = slow (crosses core, oversubscribed)
```

### Placement Groups (AWS)

```yaml
# AWS: Use Cluster Placement Group for lowest latency
# All instances in same physical rack

# Create placement group:
# aws ec2 create-placement-group --group-name ml-training --strategy cluster

# In Karpenter provisioner:
apiVersion: karpenter.sh/v1beta1
kind: NodePool
spec:
  template:
    spec:
      requirements:
      - key: "node.kubernetes.io/instance-type"
        operator: In
        values: ["p5.48xlarge"]
      nodeClassRef:
        name: gpu-nodes

---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: gpu-nodes
spec:
  instanceProfile: "gpu-node-profile"
  placementGroup: "ml-training"    # ← Force same rack!
```

## Debugging Topology Issues

```bash
# 1. Check GPU topology
nvidia-smi topo -m

# 2. Check NUMA affinity
numactl --hardware
nvidia-smi topo -p    # PCIe topology details
lstopo                 # Visual topology (install hwloc)

# 3. Check which NUMA node GPU is on
cat /sys/bus/pci/devices/0000:XX:00.0/numa_node

# 4. Check InfiniBand HCA NUMA affinity
cat /sys/class/infiniband/mlx5_0/device/numa_node

# 5. Verify GPUDirect RDMA path
# If GPU and NIC on different NUMA → RDMA goes through CPU → slower
nvidia-smi topo -m | grep mlx  # Check GPU↔NIC connection type
# PXB = optimal, SYS = suboptimal
```

## Performance Impact Example

```
Scenario: 2 GPUs training, exchanging 1GB gradient

Same NVLink pair:
  Time: 1.7 ms (600 GB/s)

Same PCIe switch (no NVLink):
  Time: 16 ms (64 GB/s) — 10x SLOWER

Cross NUMA (different socket):
  Time: 25 ms (40 GB/s) — 15x SLOWER

→ Bad GPU allocation can make training 10-15x slower!
```

## Interview Questions

1. **Q: nvidia-smi topo -m mein SYS dikhta hai GPU aur NIC ke beech. Problem kya hai?**
   A: SYS = cross-NUMA socket. GPUDirect RDMA ka data CPU socket cross karega → extra latency. Fix: NIC ko same NUMA node pe move karo (hardware change) ya scheduler ko configure karo ki us GPU ko remote traffic na de.

2. **Q: 4-node training setup kar rahe ho. Kaise ensure karoge ki nodes physically close hain?**
   A: AWS pe Cluster Placement Group use karo — ye guarantee karta hai same rack/low-latency network. K8s mein topology spread constraints + node affinity use karo.

3. **Q: Kubernetes Topology Manager ka "single-numa-node" policy kab use karte ho?**
   A: Jab GPU + NIC + CPU sab same NUMA pe chahiye (ML training). Strict policy hai — pod reject ho jayega agar alignment possible nahi. Use for performance-critical training, not for general workloads.

4. **Q: 8-GPU node pe 2 pods deploy karne hain, each with 4 GPUs. Kaise ensure karoge optimal allocation?**
   A: Topology-aware device plugin configure karo → it will allocate GPU0-3 (same NVSwitch group) to pod 1 and GPU4-7 to pod 2. Verify with nvidia-smi topo ki allocated GPUs NVLink-connected hain.
