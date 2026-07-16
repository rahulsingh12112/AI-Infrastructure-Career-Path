# Week 2: Kubernetes for ML

## Why This Matters
You already know K8s. Adding ML workload management = deadly combo. Cisco Bangalore is hiring "Kubernetes Platform Engineer - AI Infrastructure (8+ yrs)"

## Real Job Example
**Cisco — AI Infrastructure (8+ Years), Bangalore:**

> "Lead technical direction, ensuring performance, reliability, and scalability of AI systems"

---

## Topics

### NVIDIA GPU Operator
* Automates GPU driver & runtime on K8s nodes
* NVIDIA device plugin, DCGM exporter
* MIG (Multi-Instance GPU) configuration

### MIG vs MPS — When to Use What
* **MIG (Multi-Instance GPU):** Hardware-level partitioning (A100/H100 only)
  * 1 A100 → up to 7 isolated GPU instances
  * Best for: Multi-tenant inference, different users sharing 1 GPU
  * Each partition has dedicated memory & compute — full isolation
* **MPS (Multi-Process Service):** Software-level GPU sharing
  * Multiple processes share same GPU simultaneously
  * Best for: Small models that don't fully utilize GPU
  * No memory isolation — one process can affect another
* **When to use which:**
  * Production multi-tenant → MIG
  * Dev/test environment → MPS
  * Large training job → Neither (give full GPU)

### KubeFlow
* ML pipeline orchestration on Kubernetes
* Jupyter notebooks, training operators, serving

### Ray on K8s
* Distributed computing framework
* Ray Serve for model serving
* KubeRay operator

### GPU Scheduling
* Resource requests/limits for nvidia.com/gpu
* Node affinity & taints for GPU nodes
* Multi-tenancy: fair sharing GPUs across teams

### Volcano / Yunikorn — Gang Scheduling
* **Problem:** Default K8s scheduler assigns GPUs one pod at a time — distributed training needs ALL pods to start together
* **Volcano Scheduler:**
  * Gang scheduling — "all or nothing" pod group scheduling
  * Queue management for ML jobs
  * Fair-share policies across teams
  * Priority-based preemption
* **Apache Yunikorn:**
  * Alternative to Volcano, used in big data + ML
  * Hierarchical queues, resource quotas
* **When needed:** Any multi-node distributed training (PyTorch DDP, Horovod)

### NCCL & Multi-Node GPU Communication
* **NCCL (NVIDIA Collective Communications Library):**
  * GPU-to-GPU communication for distributed training
  * Operations: AllReduce, AllGather, Broadcast, ReduceScatter
  * Runs over NVLink (intra-node) and InfiniBand/EFA (inter-node)
* **Why it matters on K8s:**
  * Pods on different nodes need high-speed GPU communication
  * Network topology affects NCCL performance dramatically
  * Misconfigured networking = training hangs or 10x slower
* **Key concepts:**
  * Ring AllReduce vs Tree AllReduce
  * NCCL environment variables (NCCL_SOCKET_IFNAME, NCCL_DEBUG)
  * GPUDirect RDMA — GPU talks directly to network card (bypasses CPU)

### Network Topology Awareness
* **Problem:** Not all GPU connections are equal
  * Same node, NVLink: 600 GB/s
  * Same node, PCIe: 64 GB/s
  * Cross node, InfiniBand: 400 Gb/s
  * Cross node, Ethernet: 100 Gb/s
* **Topology-aware scheduling:**
  * Place communicating pods on same node when possible
  * Use topology labels (nvidia.com/gpu.topology)
  * Prefer NVLink-connected GPUs for multi-GPU training
* **Tools:**
  * `nvidia-smi topo -m` — show GPU topology
  * Topology-aware device plugin configuration

### GPU Monitoring & Troubleshooting
* **DCGM (Data Center GPU Manager) — Deep Dive:**
  * GPU utilization, memory usage, temperature, power
  * ECC errors, XID errors (hardware failures)
  * Profiling metrics: SM occupancy, tensor core utilization
* **Common GPU issues on K8s:**
  * OOM (Out of Memory) — pod killed, training lost
  * GPU memory leak — utilization stays high after pod dies
  * NCCL timeout — distributed training hangs
  * XID 79 error — GPU fell off the bus (hardware failure)
* **Debugging commands:**
  * `nvidia-smi` — basic GPU status
  * `nvidia-smi dmon` — continuous monitoring
  * `dcgmi diag -r 3` — full GPU health check
  * `nccl-tests` — verify multi-GPU communication
* **Monitoring stack:**
  * DCGM Exporter → Prometheus → Grafana dashboards
  * Alert on: GPU memory > 90%, temperature > 80°C, ECC errors > 0

### Checkpoint & Resume on K8s
* **Problem:** Training takes days/weeks. If pod dies → all progress lost
* **Solution: Periodic checkpointing**
  * Save model state every N steps to persistent storage
  * On failure, new pod loads last checkpoint and continues
* **Implementation on K8s:**
  * PersistentVolumeClaim (PVC) for checkpoint storage
  * Training script saves to shared storage (S3, FSx, EFS)
  * Job controller detects failure → restarts pod → resumes from checkpoint
* **Tools:**
  * PyTorch `torch.save()` / `torch.load()` for model state
  * Elastic training (torchElastic / `torchrun`) — auto-recovery
  * SageMaker HyperPod — managed auto-recovery
* **Best practices:**
  * Checkpoint every 30-60 minutes
  * Keep last 3 checkpoints (rolling)
  * Use async checkpointing (don't block training)

### Storage for ML Workloads
* **The problem:** GPUs are fast, but data loading is often the bottleneck
* **Storage options comparison:**

| Storage | Speed | Use Case |
|---------|-------|----------|
| FSx for Lustre | Very fast (parallel FS) | Training data, large datasets |
| EFS | Medium | Shared notebooks, configs |
| S3 + Mountpoint | Good for sequential | Checkpoint storage, datasets |
| Local NVMe (instance store) | Fastest | Caching, temp data |
| NFS | Slow | Small shared files |

* **Key concepts:**
  * Data pipeline: S3 → FSx cache → GPU memory
  * Pre-fetch & cache strategies
  * Avoid GPU idle time waiting for data

### EKS Specifics
* Karpenter for GPU node auto-scaling
* EFA on EKS for distributed training
* FSx for Lustre (fast shared storage)

### Cost Optimization
* **Spot Instances for training:**
  * 60-90% cheaper than on-demand
  * Risk: AWS can reclaim with 2-min warning
  * Solution: Checkpoint frequently + spot interruption handler
* **Idle GPU detection:**
  * Monitor GPU utilization — alert if < 10% for 30 min
  * Auto-scale down idle nodes (Karpenter handles this)
* **Right-sizing:**
  * Don't use p5.48xlarge if p4d.24xlarge is enough
  * Profile first → pick instance type
* **Multi-tenancy:**
  * Time-slicing: Multiple users share GPU by time
  * Namespace quotas: Limit GPU per team
  * Priority classes: Training > inference > dev

---

## Hands-on Tasks
- [ ] Deploy NVIDIA GPU Operator on EKS
- [ ] Install KubeFlow on EKS
- [ ] Run distributed PyTorch training on K8s
- [ ] Set up Ray Serve for model inference
- [ ] Configure Karpenter for GPU auto-scaling
- [ ] Run NCCL all-reduce test across 2+ GPU nodes
- [ ] Simulate GPU failure + verify job auto-recovery with checkpointing
- [ ] Configure MIG partitioning on A100 (1 GPU → 7 instances) + assign workloads
- [ ] Deploy Volcano scheduler + run gang-scheduled distributed training
- [ ] Build GPU monitoring dashboard (Grafana + DCGM Exporter + Prometheus)
- [ ] Set up Spot instance interruption handling for training jobs
- [ ] Run `nvidia-smi topo -m` and schedule pods based on GPU topology

---

## Resources
* [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
* [KubeFlow on AWS](https://awslabs.github.io/kubeflow-manifests/)
* [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
* [EKS Best Practices for ML](https://aws.github.io/aws-eks-best-practices/machine-learning/)
* [Volcano Scheduler](https://volcano.sh/en/)
* [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
* [NCCL Tests (GitHub)](https://github.com/NVIDIA/nccl-tests)
* [DCGM User Guide](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/)
* [PyTorch Elastic (torchrun)](https://pytorch.org/docs/stable/elastic/run.html)
* [GPU Topology-Aware Scheduling](https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/)
* [Karpenter GPU Best Practices](https://karpenter.sh/)
