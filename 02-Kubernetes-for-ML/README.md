# Week 2: Kubernetes for ML

## Why This Matters
You already know K8s. Adding ML workload management = deadly combo.
Cisco Bangalore is hiring "Kubernetes Platform Engineer - AI Infrastructure (8+ yrs)"

## Real Job Example
**Cisco — AI Infrastructure (8+ Years), Bangalore:**
> "Lead technical direction, ensuring performance, reliability, and scalability of AI systems"

## Topics

### NVIDIA GPU Operator
- Automates GPU driver & runtime on K8s nodes
- NVIDIA device plugin, DCGM exporter
- MIG (Multi-Instance GPU) configuration

### KubeFlow
- ML pipeline orchestration on Kubernetes
- Jupyter notebooks, training operators, serving

### Ray on K8s
- Distributed computing framework
- Ray Serve for model serving
- KubeRay operator

### GPU Scheduling
- Resource requests/limits for nvidia.com/gpu
- Node affinity & taints for GPU nodes
- Multi-tenancy: fair sharing GPUs across teams

### EKS Specifics
- Karpenter for GPU node auto-scaling
- EFA on EKS for distributed training
- FSx for Lustre (fast shared storage)

## Hands-on Tasks
- [ ] Deploy NVIDIA GPU Operator on EKS
- [ ] Install KubeFlow on EKS
- [ ] Run distributed PyTorch training on K8s
- [ ] Set up Ray Serve for model inference
- [ ] Configure Karpenter for GPU auto-scaling

## Resources
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [KubeFlow on AWS](https://awslabs.github.io/kubeflow-manifests/)
- [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [EKS Best Practices for ML](https://aws.github.io/aws-eks-best-practices/machine-learning/)
