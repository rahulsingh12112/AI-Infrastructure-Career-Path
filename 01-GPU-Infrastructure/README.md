# Week 1: GPU Infrastructure on AWS

## Why This Matters
GPU cluster networking is a RARE skill. Your 8 years networking experience directly applies here.
NVIDIA India pays avg ₹83.4 LPA for AI Engineers. They need people who understand GPU networking.

## Real Job Example
**NVIDIA — Senior AI Infrastructure Engineer, DGX Cloud:**
> "Distributed systems, automation, Kubernetes, and GPU infrastructure, ensuring reliability, performance, and scalability for AI training and inference services across production environments globally at enterprise scale."

## Topics

### EC2 GPU Instances
| Instance | GPU | Use Case |
|----------|-----|----------|
| P4d | 8x A100 (40GB) | Training |
| P5 | 8x H100 (80GB) | Large model training |
| G5 | NVIDIA A10G | Inference |
| Inf2 | AWS Inferentia2 | Cost-efficient inference |
| Trn1 | AWS Trainium | Training (AWS custom chip) |

### Key Networking Concepts for GPU (Your expertise!)
- **EFA (Elastic Fabric Adapter):** Low-latency networking for multi-node training
- **NCCL:** NVIDIA's library for multi-GPU communication
- **Placement Groups:** Cluster placement for lowest latency between GPU nodes
- **NVLink/NVSwitch:** Intra-node GPU-to-GPU communication
- **RDMA:** Remote Direct Memory Access for GPU clusters

### Hands-on Tasks
- [ ] Launch P4d instance and run GPU benchmark
- [ ] Configure EFA for distributed training
- [ ] Set up multi-node training with NCCL
- [ ] Compare cost: Inferentia vs GPU for inference
- [ ] Design network topology for GPU cluster
- [ ] Spot instance strategy for training jobs

## Resources
- [AWS GPU instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [EFA documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [SageMaker distributed training](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html)
- [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)
