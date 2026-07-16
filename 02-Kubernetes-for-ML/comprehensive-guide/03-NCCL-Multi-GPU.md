# 03 — NCCL & Multi-Node GPU Communication

## Why This is Critical

Distributed training = multiple GPUs working together on same model.
GPUs need to exchange gradients after every training step.
If communication is slow → GPUs sit idle waiting → training speed drops.

**NCCL is the #1 factor that determines distributed training performance.**

## What is NCCL?

NCCL (NVIDIA Collective Communications Library) = Library that handles GPU-to-GPU data transfer.

```
Without NCCL:
GPU 0 → CPU → Network → CPU → GPU 1  (slow, 5+ hops)

With NCCL:
GPU 0 → NVLink/InfiniBand → GPU 1    (fast, direct)
```

## Collective Operations

```
┌─────────────────────────────────────────────────────────┐
│                  NCCL Operations                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. AllReduce (MOST IMPORTANT for training)              │
│     Every GPU has partial gradients → after AllReduce,   │
│     every GPU has SUM of all gradients                   │
│                                                          │
│     Before:  GPU0=[1,2] GPU1=[3,4] GPU2=[5,6]          │
│     After:   GPU0=[9,12] GPU1=[9,12] GPU2=[9,12]       │
│     (1+3+5=9, 2+4+6=12)                                │
│                                                          │
│  2. AllGather                                            │
│     Each GPU has a piece → after, all have everything    │
│     Before:  GPU0=[A] GPU1=[B] GPU2=[C]                 │
│     After:   GPU0=[A,B,C] GPU1=[A,B,C] GPU2=[A,B,C]   │
│                                                          │
│  3. ReduceScatter                                        │
│     Reduce + distribute different pieces to each GPU     │
│     Before:  GPU0=[1,2,3] GPU1=[4,5,6] GPU2=[7,8,9]   │
│     After:   GPU0=[12] GPU1=[15] GPU2=[18]             │
│     (piece 0: 1+4+7=12, piece 1: 2+5+8=15, etc.)      │
│                                                          │
│  4. Broadcast                                            │
│     One GPU sends data to all others                     │
│     Before:  GPU0=[DATA] GPU1=[?] GPU2=[?]             │
│     After:   GPU0=[DATA] GPU1=[DATA] GPU2=[DATA]       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## AllReduce Algorithms

### Ring AllReduce
```
4 GPUs in a ring:

Step 1: Each GPU sends chunk to neighbor
GPU0 ──chunk0──→ GPU1 ──chunk1──→ GPU2 ──chunk2──→ GPU3 ──chunk3──→ GPU0

Step 2: Repeat N-1 times (reduce phase)
Step 3: Repeat N-1 times (gather phase)

Total data transferred per GPU: 2 × (N-1)/N × data_size
Time: O(data_size)  — scales well with more GPUs for large messages
```

### Tree AllReduce
```
       GPU0 (root)
      /    \
   GPU1    GPU2
   /
GPU3

Step 1: Leaves → root (reduce up)
Step 2: Root → leaves (broadcast down)

Time: O(log N) steps but each step transfers full data
Better for: Small messages, low latency needed
```

### NCCL Auto-selects Algorithm Based On:
```
Message size < 256KB  → Tree (low latency)
Message size > 256KB  → Ring (high bandwidth)
Many nodes            → Double binary tree
NVLink available      → NVLink-optimized paths
```

## Communication Paths (Speed Hierarchy)

```
Intra-node (GPUs on same machine):
┌─────────────────────────────────────────────┐
│  NVLink (A100): 600 GB/s bidirectional      │  ← FASTEST
│  NVLink (H100): 900 GB/s bidirectional      │
│  PCIe Gen4:     64 GB/s                     │  ← 10x slower
│  PCIe Gen5:     128 GB/s                    │
└─────────────────────────────────────────────┘

Inter-node (GPUs on different machines):
┌─────────────────────────────────────────────┐
│  InfiniBand NDR:  400 Gb/s (50 GB/s)       │  ← Best for training
│  InfiniBand HDR:  200 Gb/s (25 GB/s)       │
│  AWS EFA:         400 Gb/s                   │
│  RoCE v2:        100-400 Gb/s               │
│  TCP/Ethernet:   100 Gb/s                    │  ← SLOWEST
└─────────────────────────────────────────────┘
```

## GPUDirect Technologies

```
┌────────────────────────────────────────────────────────┐
│ GPUDirect — Eliminate CPU from data path                │
├────────────────────────────────────────────────────────┤
│                                                         │
│ 1. GPUDirect P2P (Peer-to-Peer):                       │
│    GPU ←→ GPU via NVLink (same node)                    │
│    No CPU copy needed                                   │
│                                                         │
│ 2. GPUDirect RDMA:                                      │
│    GPU ←→ Network Card (InfiniBand/EFA)                 │
│    Data goes directly from GPU memory to network        │
│    CPU completely bypassed!                              │
│                                                         │
│ Normal path:  GPU → CPU RAM → NIC → Network            │
│ RDMA path:    GPU → NIC → Network  (skips CPU!)        │
│                                                         │
│ 3. GPUDirect Storage:                                   │
│    GPU ←→ NVMe SSD directly                            │
│    For loading data without CPU bottleneck              │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## NCCL on Kubernetes

### Environment Variables (Critical for Performance)

```bash
# Network interface selection
NCCL_SOCKET_IFNAME=eth0          # Which network interface to use
NCCL_IB_HCA=mlx5                 # Which InfiniBand HCA to use

# Debugging
NCCL_DEBUG=INFO                  # Enable NCCL debug logs
NCCL_DEBUG_SUBSYS=ALL            # All subsystems

# Performance tuning
NCCL_ALGO=Ring                   # Force ring algorithm
NCCL_PROTO=Simple                # Protocol: Simple, LL, LL128
NCCL_CROSS_NIC=1                 # Allow cross-NIC communication
NCCL_NET_GDR_LEVEL=5             # GPUDirect RDMA level
NCCL_P2P_LEVEL=NVL               # P2P level (NVL=NVLink)
NCCL_SHM_DISABLE=0               # Shared memory for intra-node

# AWS EFA specific
FI_PROVIDER=efa                  # Use EFA provider
FI_EFA_USE_DEVICE_RDMA=1         # Enable RDMA on EFA
NCCL_PROTO=Simple                # EFA works best with Simple
```

### Pod Spec for Multi-Node Training

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nccl-worker-0
spec:
  containers:
  - name: training
    image: nvcr.io/nvidia/pytorch:24.01-py3
    env:
    - name: NCCL_SOCKET_IFNAME
      value: "eth0"
    - name: NCCL_DEBUG
      value: "INFO"
    - name: NCCL_IB_DISABLE
      value: "0"          # Enable InfiniBand
    resources:
      limits:
        nvidia.com/gpu: 8
        vpc.amazonaws.com/efa: 4   # EFA interfaces (AWS)
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi     # Large shared memory for NCCL
```

## NCCL Tests — Benchmarking

```bash
# Install nccl-tests
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu

# Run all-reduce test (most important)
mpirun -np 8 --hostfile hosts \
  ./build/all_reduce_perf -b 1M -e 1G -f 2 -g 1

# Output interpretation:
#  size(B)  count   type   redop  time(us)  algbw(GB/s)  busbw(GB/s)
#  1048576  262144  float  sum    45.2      23.2         43.5
#
# algbw = algorithm bandwidth (raw throughput)
# busbw = bus bandwidth (actual bandwidth utilization)
# For 8 GPUs with NVLink: busbw should be ~500-550 GB/s intra-node
# For inter-node with InfiniBand 400G: busbw should be ~40-45 GB/s
```

### Expected Performance (A100 DGX — 8 GPUs per node)

```
Intra-node (8 GPUs, NVLink):
Message Size │ Bus Bandwidth
1 MB         │ ~200 GB/s
64 MB        │ ~500 GB/s
1 GB         │ ~550 GB/s

Inter-node (16 GPUs, 2 nodes, InfiniBand HDR):
Message Size │ Bus Bandwidth
1 MB         │ ~5 GB/s
64 MB        │ ~22 GB/s
1 GB         │ ~24 GB/s
```

## Common NCCL Issues on K8s

### Issue 1: NCCL Timeout (Training Hangs)
```
Error: "Watchdog caught collective operation timeout"

Causes:
├── Network misconfiguration (wrong interface)
├── One GPU is slower (straggler)
├── InfiniBand port down on one node
├── Firewall blocking NCCL ports
└── Shared memory too small (/dev/shm)

Fix:
├── Set NCCL_SOCKET_IFNAME correctly
├── Increase NCCL_TIMEOUT (default 1800s)
├── Check: ibstat (InfiniBand status)
├── Ensure /dev/shm is large enough (64Gi+)
└── Run nccl-tests to isolate the problem
```

### Issue 2: Slow Communication
```
Symptom: busbw much lower than expected

Causes:
├── GPUDirect RDMA not enabled
├── Wrong NCCL algorithm selected
├── PCIe used instead of NVLink
├── Network congestion
└── EFA not properly configured

Debug:
├── NCCL_DEBUG=INFO → check which path NCCL chose
├── nvidia-smi topo -m → verify NVLink connections
├── ibstat → verify IB link speed
└── Check NCCL_NET_GDR_LEVEL setting
```

### Issue 3: NCCL Init Failure
```
Error: "NCCL error: unhandled system error"

Causes:
├── Pods can't reach each other (DNS/network policy)
├── Wrong number of ranks configured
├── NCCL_SOCKET_IFNAME pointing to wrong interface
└── GPU driver mismatch between nodes

Fix:
├── Verify pod-to-pod connectivity (ping)
├── Check MASTER_ADDR and MASTER_PORT
├── Ensure same CUDA/NCCL version on all nodes
└── Use headless service for stable DNS
```

## Real-World: Distributed Training on K8s

```yaml
# PyTorch Distributed Training Job (PyTorchJob CRD via Kubeflow)
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: distributed-training
spec:
  nprocPerNode: "8"       # 8 GPUs per node
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      template:
        spec:
          containers:
          - name: pytorch
            image: my-training:latest
            env:
            - name: NCCL_DEBUG
              value: "WARN"
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            resources:
              limits:
                nvidia.com/gpu: 8
                vpc.amazonaws.com/efa: 4
            volumeMounts:
            - name: shm
              mountPath: /dev/shm
          volumes:
          - name: shm
            emptyDir:
              medium: Memory
              sizeLimit: 64Gi
    Worker:
      replicas: 3         # 3 more nodes = 4 total = 32 GPUs
      template:
        spec:
          containers:
          - name: pytorch
            image: my-training:latest
            env:
            - name: NCCL_DEBUG
              value: "WARN"
            - name: NCCL_SOCKET_IFNAME
              value: "eth0"
            resources:
              limits:
                nvidia.com/gpu: 8
                vpc.amazonaws.com/efa: 4
            volumeMounts:
            - name: shm
              mountPath: /dev/shm
          volumes:
          - name: shm
            emptyDir:
              medium: Memory
              sizeLimit: 64Gi
```

## Interview Questions

1. **Q: AllReduce kyun zaroori hai training mein?**
   A: Data parallel training mein har GPU apne data batch pe gradients calculate karta hai. AllReduce se saare GPUs ke gradients average hote hain → sab GPUs ke paas same updated weights aate hain → next step sab synchronized rehte hain.

2. **Q: NCCL timeout ho raha hai distributed training mein. Kaise debug karoge?**
   A: NCCL_DEBUG=INFO set karo → logs mein dekhh konsa rank stuck hai → us node pe ibstat check karo (IB port down?) → nccl-tests run karo us node se isolate mein → /dev/shm size check karo → network policy/firewall check karo.

3. **Q: GPUDirect RDMA se kitna farak padta hai?**
   A: Without RDMA: GPU→CPU RAM→NIC = 2 extra copies + CPU overhead. With RDMA: GPU→NIC directly = ~30-50% better inter-node bandwidth, significantly lower latency.

4. **Q: 64 GPU training (8 nodes × 8 GPUs). NCCL kaise optimize karoge?**
   A: Intra-node NVLink use karo (P2P_LEVEL=NVL), inter-node InfiniBand with GPUDirect RDMA (NET_GDR_LEVEL=5), hierarchical allreduce (pehle intra-node, fir inter-node), large /dev/shm (64Gi+), EFA 4x interfaces per node.
