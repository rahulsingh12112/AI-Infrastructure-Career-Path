# 03 — SageMaker HyperPod (GPU Cluster Management)

## What is HyperPod?

Managed GPU cluster for long-running distributed training with automatic fault recovery.

```
Problem without HyperPod:
├── GPU fails mid-training (after 5 days) → restart from scratch
├── Node goes down → manual SSH, debug, replace
├── Cluster health? → You monitor manually
├── Driver update? → Node by node, downtime
└── Multi-day training = constant babysitting

With HyperPod:
├── GPU fails → auto-detected → node replaced → training resumes
├── Health monitoring → proactive (catch issues BEFORE failure)
├── Deep health checks → ECC errors, NVLink degradation
├── Cluster managed as single unit → no SSH needed
└── Train for weeks unattended
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SageMaker HyperPod                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Control Plane (AWS Managed):                                    │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Cluster Manager                                      │       │
│  │  ├── Health Monitor (continuous GPU/node checks)      │       │
│  │  ├── Auto-Recovery (replace failed nodes)             │       │
│  │  ├── Lifecycle Manager (setup/teardown)               │       │
│  │  └── Task Governance (job scheduling, quotas)         │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  Worker Nodes:                                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Node 0   │ │ Node 1   │ │ Node 2   │ │ Node 3   │          │
│  │ 8×H100   │ │ 8×H100   │ │ 8×H100   │ │ 8×H100   │          │
│  │ EFA×32   │ │ EFA×32   │ │ EFA×32   │ │ EFA×32   │          │
│  │ Health   │ │ Health   │ │ Health   │ │ Health   │           │
│  │ Agent    │ │ Agent    │ │ Agent    │ │ Agent    │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                  │
│  Orchestrator:                                                   │
│  ├── Option 1: Slurm (HPC-style job scheduling)                 │
│  └── Option 2: Amazon EKS (Kubernetes-based)                    │
│                                                                  │
│  Storage:                                                        │
│  ├── FSx for Lustre (training data, checkpoints)                │
│  └── S3 (model artifacts, datasets)                             │
│                                                                  │
│  Networking:                                                     │
│  ├── EFA (GPU-to-GPU, NCCL)                                    │
│  └── Cluster placement group (low latency)                      │
└─────────────────────────────────────────────────────────────────┘
```

## HyperPod Cluster Creation

### Using AWS CLI

```bash
# Create HyperPod cluster
aws sagemaker create-cluster \
  --cluster-name "my-training-cluster" \
  --instance-groups '[
    {
      "InstanceGroupName": "controller",
      "InstanceType": "ml.m5.xlarge",
      "InstanceCount": 1,
      "LifeCycleConfig": {
        "SourceS3Uri": "s3://my-bucket/lifecycle-scripts/",
        "OnCreate": "setup_controller.sh"
      },
      "ExecutionRole": "arn:aws:iam::123456789:role/HyperPodRole"
    },
    {
      "InstanceGroupName": "gpu-workers",
      "InstanceType": "ml.p5.48xlarge",
      "InstanceCount": 8,
      "LifeCycleConfig": {
        "SourceS3Uri": "s3://my-bucket/lifecycle-scripts/",
        "OnCreate": "setup_worker.sh"
      },
      "ExecutionRole": "arn:aws:iam::123456789:role/HyperPodRole"
    }
  ]' \
  --vpc-config '{
    "SecurityGroupIds": ["sg-xxx"],
    "Subnets": ["subnet-xxx"]
  }'
```

### Lifecycle Scripts

```bash
#!/bin/bash
# setup_worker.sh — runs on each worker node at creation

# Install NVIDIA drivers + CUDA
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 cuda-toolkit-12-2

# Install EFA
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
tar -xf aws-efa-installer-latest.tar.gz
cd aws-efa-installer && sudo ./efa_installer.sh -y

# Install NCCL
sudo apt-get install -y libnccl2 libnccl-dev

# Install Slurm worker
sudo apt-get install -y slurmd slurm-client

# Mount FSx
sudo mount -t lustre fs-xxx.fsx.us-east-1.amazonaws.com@tcp:/fsx /shared

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Pull training container
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# Start health monitoring agent
sudo systemctl start sagemaker-health-monitor
```

## HyperPod with Slurm

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 HyperPod + Slurm                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Controller Node (ml.m5.xlarge):                        │
│  ├── Slurm Controller (slurmctld)                       │
│  ├── Slurm Database (slurmdbd)                          │
│  └── Login access for users                             │
│                                                          │
│  Login Node (optional):                                  │
│  └── Users SSH here to submit jobs                      │
│                                                          │
│  Worker Nodes (ml.p5.48xlarge × 8):                     │
│  ├── Slurm Worker (slurmd)                              │
│  ├── 8 × H100 GPUs                                     │
│  ├── EFA networking                                     │
│  └── Shared FSx mount (/shared)                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Slurm Job Submission

```bash
# SSH into login/controller node
aws ssm start-session --target sagemaker-cluster:my-training-cluster_controller-i-xxx

# Submit training job
sbatch train.slurm
```

### Slurm Job Script (train.slurm)

```bash
#!/bin/bash
#SBATCH --job-name=llm-training
#SBATCH --nodes=4                    # 4 nodes
#SBATCH --ntasks-per-node=8          # 8 GPUs per node
#SBATCH --gres=gpu:8                 # Request 8 GPUs per node
#SBATCH --exclusive                  # No sharing
#SBATCH --output=/shared/logs/%j.out
#SBATCH --error=/shared/logs/%j.err
#SBATCH --time=72:00:00              # Max 72 hours
#SBATCH --partition=gpu

# Environment
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ens
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run distributed training
srun torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$(hostname):29400 \
  /shared/code/train.py \
  --model meta-llama/Llama-2-70b-hf \
  --data_path /shared/data/train/ \
  --checkpoint_dir /shared/checkpoints/ \
  --batch_size 4 \
  --gradient_accumulation_steps 8
```

### Slurm Commands

```bash
# Submit job
sbatch train.slurm

# Check job status
squeue -u $USER

# Check all jobs
squeue -l

# Cancel job
scancel <job_id>

# Interactive session (for debugging)
srun --nodes=1 --gres=gpu:8 --pty bash

# Check node status
sinfo -N -l

# Check GPU status across cluster
srun --nodes=4 nvidia-smi

# Job history
sacct -j <job_id> --format=JobID,Start,End,Elapsed,MaxRSS,ExitCode
```

## HyperPod with EKS

```
┌─────────────────────────────────────────────────────────┐
│               HyperPod + EKS                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  EKS Control Plane (AWS managed):                        │
│  └── API Server, etcd, controllers                      │
│                                                          │
│  HyperPod Worker Nodes:                                  │
│  ├── Registered as EKS nodes                            │
│  ├── GPU Operator installed                              │
│  ├── EFA device plugin                                   │
│  ├── Health monitoring agent                             │
│  └── Auto-recovery on failure                           │
│                                                          │
│  Benefits:                                               │
│  ├── Kubernetes-native (PyTorchJob, KubeFlow)           │
│  ├── HyperPod health monitoring + auto-replace          │
│  ├── Training Operator for job auto-resume              │
│  └── Best of both: K8s flexibility + HyperPod resiliency│
└─────────────────────────────────────────────────────────┘
```

### HyperPod Training Operator (EKS)

```yaml
# Training job that auto-resumes on node failure
apiVersion: sagemaker.aws.amazon.com/v1
kind: TrainingJob
metadata:
  name: llm-finetuning
spec:
  clusterConfig:
    clusterName: my-hyperpod-cluster
  trainingConfig:
    image: nvcr.io/nvidia/pytorch:24.01-py3
    command:
    - torchrun
    - --nnodes=4
    - --nproc_per_node=8
    - train.py
    resources:
      instanceType: ml.p5.48xlarge
      instanceCount: 4
    checkpointConfig:
      s3Uri: s3://my-bucket/checkpoints/
      localPath: /opt/ml/checkpoints
    # Auto-resume: If node fails, operator restarts from checkpoint
    restartPolicy:
      maxRestarts: 5
      backoffLimit: 3
```

## Auto-Recovery (Key Feature)

### How It Works

```
Normal operation:
  Node health checks every 30 seconds:
  ├── GPU health (DCGM diagnostics)
  ├── ECC error count
  ├── NVLink status
  ├── EFA connectivity
  ├── Disk health
  └── Network reachability

Failure detected:
  ┌─────────────────────────────────────────────┐
  │ 1. Health agent detects GPU XID error        │
  │ 2. Node marked unhealthy                     │
  │ 3. Training checkpoints saved (if running)   │
  │ 4. Failed node cordoned                      │
  │ 5. New node provisioned (same instance type) │
  │ 6. Lifecycle scripts run (setup environment) │
  │ 7. Node joins cluster                        │
  │ 8. Training resumes from last checkpoint     │
  │                                              │
  │ Total downtime: ~5-15 minutes                │
  └─────────────────────────────────────────────┘
```

### Recovery Scenarios

```
Scenario              │ Recovery Action              │ Downtime
──────────────────────┼──────────────────────────────┼──────────
GPU ECC error         │ Node replaced automatically  │ ~10 min
GPU fell off bus      │ Node replaced automatically  │ ~10 min
NVLink failure        │ Node replaced automatically  │ ~10 min
Node unreachable      │ Node replaced automatically  │ ~15 min
Disk failure          │ Node replaced automatically  │ ~10 min
Software crash        │ Process restarted (no replace)│ ~1 min
Network blip          │ NCCL retry (no replace)      │ ~seconds
EFA adapter failure   │ Node replaced automatically  │ ~10 min
```

## Task Governance (Multi-Tenancy)

```python
# HyperPod Task Governance — fair sharing GPU cluster across teams

# Create compute allocation
aws sagemaker create-compute-quota \
  --cluster-name "my-cluster" \
  --compute-quota-config '{
    "ComputeQuotaId": "research-team",
    "ComputeQuotaTarget": {
      "FairShareWeight": 4
    },
    "ResourceSharingConfig": {
      "Strategy": "Lend"    # Lend idle resources to others
    },
    "PreemptionConfig": {
      "AllowedPreemptors": ["production-team"]  # Production can preempt research
    },
    "ResourceLimits": {
      "MaxInstances": 8,           # Max 8 nodes
      "MaxGPUs": 64                # Max 64 GPUs
    }
  }'
```

### Governance Policies

```
Team Allocation Example (16-node cluster = 128 GPUs):

Team          │ Weight │ Guarantee │ Max    │ Preemptable?│ Lend Idle?
──────────────┼────────┼───────────┼────────┼─────────────┼────────────
Production    │ 4      │ 32 GPUs   │ 64 GPUs│ No          │ No
Research      │ 3      │ 24 GPUs   │ 96 GPUs│ By Prod     │ Yes
Development   │ 1      │ 0 GPUs    │ 48 GPUs│ By all      │ Yes

Behavior:
├── Idle cluster: Dev can use up to 48 GPUs (from Research+Dev idle)
├── Research submits: Dev GPUs reclaimed, Research gets up to 96
├── Production submits: Research preempted if needed, Production guaranteed 32
└── All resources used: Fair-share by weight (4:3:1 ratio)
```

## UltraServers (NVL72)

```
What is it?
├── 18 instances × 4 GPUs = 72 NVIDIA Blackwell GPUs
├── ALL connected via NVLink (not just within-node!)
├── Single NVLink domain = 72 GPUs can communicate at NVLink speed
├── No cross-node networking bottleneck
└── For trillion-parameter models

Architecture:
┌─────────────────────────────────────────────────────────┐
│              NVL72 UltraServer                            │
│                                                          │
│  ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐               │
│  │Inst 0│ │Inst 1│ │Inst 2│ ... │Inst 17│               │
│  │4 GPUs│ │4 GPUs│ │4 GPUs│     │4 GPUs│               │
│  └──┬───┘ └──┬───┘ └──┬───┘     └──┬───┘               │
│     │        │        │            │                     │
│  ═══╪════════╪════════╪════════════╪═══ NVLink Fabric    │
│     │        │        │            │                     │
│  ALL 72 GPUs in ONE NVLink domain                       │
│  = 900 GB/s any-to-any (no network hop!)                │
│                                                          │
│  Scheduling:                                             │
│  ├── HyperPod fills 1 UltraServer completely first      │
│  ├── If need 14 instances → all from UltraServer 0      │
│  ├── If need 20 → 18 from US0 + 2 from US1             │
│  └── Topology-aware: minimize cross-UltraServer comms   │
└─────────────────────────────────────────────────────────┘
```

## Monitoring & Observability

```bash
# Cluster status
aws sagemaker describe-cluster --cluster-name my-cluster

# Node health
aws sagemaker list-cluster-nodes --cluster-name my-cluster

# Node details
aws sagemaker describe-cluster-node \
  --cluster-name my-cluster \
  --node-id i-xxxx

# CloudWatch metrics:
# - GPUUtilization
# - GPUMemoryUtilization
# - DiskUtilization
# - NetworkIn/Out
# - HealthCheckFailures

# CloudWatch Logs:
# /aws/sagemaker/HyperPod/<cluster-name>/health
# /aws/sagemaker/HyperPod/<cluster-name>/lifecycle
```

## Cost Optimization with HyperPod

```
Pricing: HyperPod uses Training Plans (reserved capacity)

Option                │ Discount  │ Commitment  │ Best For
──────────────────────┼───────────┼─────────────┼──────────────
On-Demand             │ 0%        │ None        │ Short experiments
Training Plans (30d)  │ ~20-30%   │ 30 days     │ Medium training runs
Training Plans (90d)  │ ~40-50%   │ 90 days     │ Long-term research
Capacity Blocks       │ ~30-40%   │ Fixed window│ Planned training

Cost example (8-node p5.48xlarge cluster):
On-Demand:  8 × $98.32/hr = $786/hr = $18,864/day
30-day plan: ~$550/hr = ~$13,200/day (save ~$5,600/day!)
```

## HyperPod vs Self-Managed EKS

```
Feature              │ HyperPod              │ Self-Managed EKS
─────────────────────┼────────────────────────┼──────────────────────
Auto-recovery        │ Built-in (automatic)  │ DIY (write controllers)
Health monitoring    │ Deep GPU-level         │ Basic (node-level)
Node replacement    │ Automatic (~10 min)    │ Manual or Karpenter
GPU diagnostics     │ DCGM built-in          │ Install yourself
Networking          │ EFA optimized          │ Configure yourself
Storage             │ FSx integrated         │ Setup CSI drivers
Job scheduling      │ Slurm OR EKS           │ K8s only
Cost                │ Same instance cost     │ Same instance cost
                    │ + no management overhead│ + your engineer time
Setup time          │ Hours                  │ Days-Weeks
Best for            │ Long-running training  │ Mixed workloads
```

## Interview Questions

1. **Q: HyperPod pe 8-node training chal rahi hai. Node 3 ka GPU fail hua. Kya hoga?**
   A: (1) Health agent detects GPU error (XID/ECC), (2) Node 3 marked unhealthy, (3) If training job running → NCCL timeout on other nodes, (4) HyperPod provisions replacement node (~5 min), (5) New node runs lifecycle scripts (driver, CUDA, EFA setup ~3 min), (6) Node joins cluster, (7) Training framework (torchrun) restarts from last checkpoint. Total recovery: ~10-15 min.

2. **Q: Slurm vs EKS orchestrator — kab kya choose karoge?**
   A: Slurm: HPC-style workloads, research teams familiar with Slurm, simple job scheduling, bare-metal feel. EKS: Already have K8s infrastructure, want KubeFlow/Ray integration, containerized workflows, microservices alongside training. Most ML research teams prefer Slurm (simpler), platform teams prefer EKS (standard tooling).

3. **Q: 128-GPU cluster pe 3 teams share kar rahi hain. Governance kaise set up karoge?**
   A: Task Governance with compute quotas: (1) Production: weight=4, guarantee=32 GPUs, non-preemptable, (2) Research: weight=3, guarantee=24, preemptable by production, lending enabled, (3) Dev: weight=1, no guarantee, preemptable, lending enabled. Fair-share scheduler distributes idle GPUs by weight ratio.

4. **Q: UltraServer (NVL72) ka benefit kya hai over regular p5 instances?**
   A: Regular p5: 8 GPUs per node connected via NVLink, cross-node = EFA (50 GB/s). NVL72: 72 GPUs ALL in one NVLink domain (900 GB/s any-to-any). For trillion-param models, TP degree can be 72 (not limited to 8). No cross-node bottleneck for AllReduce. Entire model fits in unified GPU memory pool. 3-5x faster for models that need frequent all-to-all communication.
