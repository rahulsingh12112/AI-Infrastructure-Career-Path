# 11 — AWS EKS for ML (Karpenter, EFA, FSx)

## EKS for ML — Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EKS ML Cluster                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Control Plane (AWS Managed)                                 │
│  ├── API Server                                              │
│  ├── etcd                                                    │
│  └── Controllers                                             │
│                                                              │
│  Node Pools:                                                 │
│  ┌────────────────────────────────────────────────────┐     │
│  │ System Nodes (m5.xlarge) — CoreDNS, monitoring     │     │
│  └────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Training Nodes (p5.48xlarge) — 8× H100 + EFA      │     │
│  │ Managed by: Karpenter                               │     │
│  │ Taint: workload=training:NoSchedule                 │     │
│  └────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────┐     │
│  │ Inference Nodes (g5.xlarge) — 1× A10G             │     │
│  │ Managed by: Karpenter                               │     │
│  │ Taint: workload=inference:NoSchedule                │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  Storage:                                                    │
│  ├── FSx for Lustre (training data + checkpoints)           │
│  ├── EFS (shared configs, notebooks)                        │
│  └── S3 (data lake, model registry)                         │
│                                                              │
│  Networking:                                                 │
│  ├── EFA (Elastic Fabric Adapter) for GPU-to-GPU            │
│  ├── VPC with placement groups                              │
│  └── Pod-level security groups                              │
│                                                              │
│  Add-ons:                                                    │
│  ├── NVIDIA GPU Operator                                     │
│  ├── EFA Device Plugin                                       │
│  ├── FSx CSI Driver                                         │
│  ├── Karpenter                                              │
│  └── Volcano Scheduler                                       │
└─────────────────────────────────────────────────────────────┘
```

## Karpenter for GPU Auto-Scaling

### What is Karpenter?
AWS-native node autoscaler. Faster and more flexible than Cluster Autoscaler.

```
Cluster Autoscaler vs Karpenter:
├── CA: Pre-defined node groups, slow (~5 min), rigid
├── Karpenter: No node groups, fast (~90 sec), flexible
│   ├── Can pick optimal instance type per pod
│   ├── Consolidates underutilized nodes
│   └── Handles spot interruptions natively
```

### Karpenter NodePool for Training

```yaml
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-training
spec:
  template:
    metadata:
      labels:
        workload-type: training
    spec:
      requirements:
      - key: "node.kubernetes.io/instance-type"
        operator: In
        values:
        - "p5.48xlarge"       # 8× H100
        - "p4d.24xlarge"      # 8× A100
      - key: "karpenter.sh/capacity-type"
        operator: In
        values: ["spot", "on-demand"]  # Try spot first
      - key: "topology.kubernetes.io/zone"
        operator: In
        values: ["us-east-1a"]  # Single AZ for placement group
      taints:
      - key: workload
        value: training
        effect: NoSchedule
      nodeClassRef:
        name: gpu-training-class
  limits:
    nvidia.com/gpu: 128         # Max 128 GPUs (16 nodes)
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 10m        # Remove idle nodes after 10 min
    expireAfter: 72h             # Replace nodes every 72h (freshness)

---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: gpu-training-class
spec:
  amiFamily: AL2
  instanceProfile: "karpenter-gpu-node"
  subnetSelectorTerms:
  - tags:
      karpenter: "true"
  securityGroupSelectorTerms:
  - tags:
      karpenter: "true"
  placement:
    groupName: "ml-training-cluster"  # ← Placement group!
  blockDeviceMappings:
  - deviceName: /dev/xvda
    ebs:
      volumeSize: 500Gi              # Large root for container images
      volumeType: gp3
      throughput: 500
      iops: 5000
  userData: |
    #!/bin/bash
    # Pre-pull large training images
    crictl pull nvcr.io/nvidia/pytorch:24.01-py3
```

### Karpenter NodePool for Inference

```yaml
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: gpu-inference
spec:
  template:
    metadata:
      labels:
        workload-type: inference
    spec:
      requirements:
      - key: "node.kubernetes.io/instance-type"
        operator: In
        values:
        - "g5.xlarge"         # 1× A10G (small models)
        - "g5.2xlarge"        # 1× A10G + more CPU
        - "g5.4xlarge"        # 1× A10G + lots of CPU
      - key: "karpenter.sh/capacity-type"
        operator: In
        values: ["spot", "on-demand"]
      taints:
      - key: workload
        value: inference
        effect: NoSchedule
      nodeClassRef:
        name: gpu-inference-class
  limits:
    nvidia.com/gpu: 32
  disruption:
    consolidationPolicy: WhenUnderutilized  # Consolidate!
    consolidateAfter: 2m
```

## EFA (Elastic Fabric Adapter)

### What is EFA?
AWS's high-performance network interface for GPU-to-GPU communication.
Like InfiniBand but cloud-native.

```
Without EFA:
  GPU → CPU → Standard NIC → TCP/IP → 100 Gbps
  High latency, limited bandwidth

With EFA:
  GPU → EFA adapter → RDMA → 400 Gbps (p5) / 400 Gbps (p4d)
  Low latency, high bandwidth, bypasses OS kernel

EFA supports:
├── NCCL (GPU collective communication)
├── MPI (traditional HPC)
├── Libfabric (low-level fabric API)
└── GPUDirect RDMA (direct GPU-to-network)
```

### EFA on EKS Setup

```yaml
# Step 1: EFA Device Plugin (DaemonSet)
# Installed via EKS add-on or Helm
helm install aws-efa-k8s-device-plugin \
  --namespace kube-system \
  https://github.com/aws/eks-charts/...

# Step 2: Pod requesting EFA
apiVersion: v1
kind: Pod
metadata:
  name: efa-training
spec:
  containers:
  - name: pytorch
    image: nvcr.io/nvidia/pytorch:24.01-py3
    env:
    - name: FI_PROVIDER
      value: "efa"
    - name: FI_EFA_USE_DEVICE_RDMA
      value: "1"
    - name: NCCL_DEBUG
      value: "INFO"
    resources:
      limits:
        nvidia.com/gpu: 8
        vpc.amazonaws.com/efa: 4     # ← 4 EFA interfaces!
        # p5.48xlarge has 32 EFA interfaces
        # p4d.24xlarge has 4 EFA interfaces
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi
```

### EFA Performance Numbers

```
Instance         │ EFA Interfaces │ Network BW  │ NCCL Inter-node BW
─────────────────┼────────────────┼─────────────┼───────────────────
p5.48xlarge      │ 32             │ 3200 Gbps   │ ~380 GB/s
p4d.24xlarge     │ 4              │ 400 Gbps    │ ~45 GB/s
p4de.24xlarge    │ 4              │ 400 Gbps    │ ~45 GB/s
trn1.32xlarge    │ 8              │ 800 Gbps    │ ~90 GB/s

# Verify EFA is working:
fi_info -p efa    # Should show EFA provider
# In NCCL logs: "Using network EFA" (not "Using network Socket")
```

### Security Groups for EFA

```yaml
# EFA requires specific security group rules
# All traffic between nodes in same SG must be allowed

# Security Group:
# Inbound:
#   - All traffic from same security group (self-referencing)
# Outbound:
#   - All traffic

# Why: EFA uses custom protocol, not just TCP/UDP
# Standard security groups break EFA if too restrictive
```

## FSx for Lustre on EKS

### Setup

```yaml
# 1. StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fsx-lustre
provisioner: fsx.csi.aws.com
parameters:
  subnetId: subnet-0123456789abcdef0
  securityGroupIds: sg-0123456789abcdef0
  deploymentType: PERSISTENT_2
  perUnitStorageThroughput: "250"        # MB/s per TiB
  fileSystemTypeVersion: "2.15"
  autoImportPolicy: NEW_CHANGED_DELETED
  s3ImportPath: s3://my-training-data/
  s3ExportPath: s3://my-training-data/exports/
volumeBindingMode: Immediate
reclaimPolicy: Delete

---
# 2. PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: fsx-lustre
  resources:
    requests:
      storage: 4800Gi     # Minimum for PERSISTENT_2

---
# 3. Pod mount
spec:
  containers:
  - name: training
    volumeMounts:
    - name: data
      mountPath: /data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: training-data
```

### FSx Best Practices for ML

```
Performance Tuning:
├── Stripe count: Set high for large files (lfs setstripe -c -1 /data)
├── Cache warming: Pre-read data before training starts
├── Import policy: AUTO (lazy load from S3 on first access)
├── Storage capacity: Larger = more throughput (baseline scales with size)
└── Per-unit throughput: 250 for training, 125 sufficient for inference

Cost Optimization:
├── Use S3 as primary storage, FSx as cache layer
├── Set data expiry policy (delete from FSx, keep in S3)
├── Shared FSx across multiple training jobs
└── Size based on active working set, not total data
```

## EKS Best Practices for ML (Complete)

### 1. Cluster Setup
```bash
# Create EKS cluster optimized for ML
eksctl create cluster \
  --name ml-cluster \
  --region us-east-1 \
  --version 1.29 \
  --with-oidc \
  --without-nodegroup

# Install add-ons
aws eks create-addon --cluster-name ml-cluster --addon-name vpc-cni
aws eks create-addon --cluster-name ml-cluster --addon-name coredns
aws eks create-addon --cluster-name ml-cluster --addon-name kube-proxy

# Install Karpenter
helm install karpenter oci://public.ecr.aws/karpenter/karpenter

# Install GPU Operator
helm install gpu-operator nvidia/gpu-operator -n gpu-operator

# Install FSx CSI Driver
helm install aws-fsx-csi-driver aws-fsx-csi-driver/aws-fsx-csi-driver

# Install EFA plugin
kubectl apply -f https://raw.githubusercontent.com/aws/eks-charts/.../efa-device-plugin.yaml

# Install Volcano
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/.../volcano.yaml
```

### 2. VPC Design for ML

```
VPC Design:
├── Private subnets (GPU nodes) — no public IP
├── Single AZ for training (placement group requires same AZ)
├── Large CIDR (/16) — EFA needs many IPs
├── No NAT Gateway bottleneck (use VPC endpoints for S3, ECR)
└── VPC Endpoints:
    ├── s3 (Gateway) — data access
    ├── ecr.api + ecr.dkr (Interface) — container pulls
    ├── sts (Interface) — IAM
    └── fsx (Interface) — FSx API calls
```

### 3. Container Image Optimization

```dockerfile
# Optimized training container
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Install only what you need
RUN pip install --no-cache-dir \
    transformers==4.37.0 \
    accelerate==0.26.0 \
    datasets==2.16.0 \
    deepspeed==0.13.0

# Pre-download model weights (avoid download at runtime)
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf')"

COPY train.py /app/train.py
WORKDIR /app

# Image size: ~25GB (with model weights pre-loaded)
# Without pre-load: Training start delayed by 10-30 min for downloads!
```

### 4. Pod Disruption Budget

```yaml
# Protect training jobs from voluntary disruptions
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: training-pdb
spec:
  maxUnavailable: 0         # NO disruption allowed during training!
  selector:
    matchLabels:
      job-type: distributed-training
```

## Complete EKS ML Stack (Install Order)

```
1. EKS Cluster + VPC
2. VPC Endpoints (S3, ECR)
3. Karpenter (node provisioning)
4. NVIDIA GPU Operator (GPU drivers/runtime)
5. EFA Device Plugin (high-speed networking)
6. FSx CSI Driver (storage)
7. Volcano Scheduler (gang scheduling)
8. Prometheus + Grafana (monitoring)
9. DCGM Exporter (GPU metrics)
10. KubeFlow / KubeRay (ML platform)
```

## Interview Questions

1. **Q: Karpenter vs Cluster Autoscaler for GPU workloads?**
   A: Karpenter is better: (1) Faster provisioning (~90s vs ~5min), (2) Can pick optimal instance type per pod, (3) Native spot interruption handling, (4) Consolidation (packs pods onto fewer nodes), (5) No need to pre-define node groups. CA is okay for simple setups but Karpenter is superior for ML.

2. **Q: p5.48xlarge pe 32 EFA interfaces hain. Pod ko kitne allocate karoge?**
   A: Depends on workload. For 8-GPU training pod: 4 EFA interfaces usually enough (each EFA = 100 Gbps, 4 = 400 Gbps > NVLink cross-node capacity). For maximum performance on p5: 32 EFA interfaces (all of them). Key: Match EFA count to actual communication pattern.

3. **Q: EFA kaam nahi kar raha — NCCL "Using network Socket" dikh raha hai. Debug?**
   A: (1) Check EFA device plugin running (kubectl get pods -n kube-system | grep efa), (2) Verify pod has vpc.amazonaws.com/efa in limits, (3) Check security group allows all traffic from self, (4) Verify FI_PROVIDER=efa env var set, (5) Run fi_info -p efa inside pod — should show EFA provider, (6) Check placement group — EFA needs same AZ.

4. **Q: Training data 50TB hai S3 pe. EKS cluster kaise access karega efficiently?**
   A: FSx for Lustre with S3 data repository association. Auto-import policy = lazy load (data fetched from S3 on first access, cached on FSx). Active working set on FSx (~5TB), rest stays in S3. Result: First epoch slightly slower, subsequent epochs = full FSx speed.
