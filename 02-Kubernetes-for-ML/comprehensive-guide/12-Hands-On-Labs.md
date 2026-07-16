# 12 — Complete Hands-On Lab Guide

## Prerequisites

```bash
# Tools needed:
├── AWS CLI configured
├── kubectl
├── eksctl
├── helm
├── nvidia-smi (on GPU nodes)
└── Python 3.10+ with PyTorch
```

---

## Lab 1: Deploy NVIDIA GPU Operator on EKS

### Objective
Set up a GPU-enabled EKS cluster from scratch.

### Steps

```bash
# 1. Create EKS cluster
eksctl create cluster \
  --name gpu-lab \
  --region us-east-1 \
  --version 1.29 \
  --without-nodegroup

# 2. Create GPU node group
eksctl create nodegroup \
  --cluster gpu-lab \
  --name gpu-nodes \
  --node-type g5.xlarge \
  --nodes 2 \
  --nodes-min 0 \
  --nodes-max 4

# 3. Install GPU Operator
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true \
  --set devicePlugin.enabled=true \
  --set dcgmExporter.enabled=true

# 4. Verify
kubectl get pods -n gpu-operator
kubectl get nodes -o json | jq '.items[].status.capacity["nvidia.com/gpu"]'

# 5. Test GPU access
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  containers:
  - name: cuda
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
  restartPolicy: Never
EOF

kubectl logs gpu-test
# Should show GPU info
```

### Verification
- [ ] GPU Operator pods all Running
- [ ] Nodes show nvidia.com/gpu in capacity
- [ ] Test pod runs nvidia-smi successfully

---

## Lab 2: Install KubeFlow on EKS

### Steps

```bash
# 1. Install KubeFlow (simplified — using manifests)
git clone https://github.com/awslabs/kubeflow-manifests.git
cd kubeflow-manifests

# 2. Install prerequisites
make install-tools
make deploy-kubeflow INSTALLATION_OPTION=kustomize

# 3. Access dashboard
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80

# 4. Create notebook with GPU
# In KubeFlow UI:
# - New Notebook → Select GPU image → Request 1 GPU → Launch
```

### Verification
- [ ] KubeFlow dashboard accessible
- [ ] Can create GPU-enabled notebook
- [ ] Can run PyTorch code with CUDA in notebook

---

## Lab 3: Distributed PyTorch Training on K8s

### Objective
Train a model across 2 nodes (4 GPUs total) using PyTorchJob.

### Training Script (train.py)

```python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def train():
    local_rank = setup()
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).cuda(local_rank)
    
    model = DDP(model, device_ids=[local_rank])
    
    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    dataset = datasets.MNIST('/data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(local_rank), target.cuda(local_rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0 and local_rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Save model (only rank 0)
    if local_rank == 0:
        torch.save(model.module.state_dict(), '/checkpoints/model.pt')
    
    dist.destroy_process_group()

if __name__ == "__main__":
    train()
```

### PyTorchJob Manifest

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: distributed-mnist
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: my-training:latest  # Image with train.py
            command: ["torchrun"]
            args:
            - "--nnodes=2"
            - "--nproc_per_node=2"
            - "--rdzv_backend=c10d"
            - "--rdzv_endpoint=$(MASTER_ADDR):$(MASTER_PORT)"
            - "train.py"
            resources:
              limits:
                nvidia.com/gpu: 2
            volumeMounts:
            - name: checkpoints
              mountPath: /checkpoints
            - name: shm
              mountPath: /dev/shm
          volumes:
          - name: checkpoints
            emptyDir: {}
          - name: shm
            emptyDir:
              medium: Memory
              sizeLimit: 8Gi
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: my-training:latest
            command: ["torchrun"]
            args:
            - "--nnodes=2"
            - "--nproc_per_node=2"
            - "--rdzv_backend=c10d"
            - "--rdzv_endpoint=$(MASTER_ADDR):$(MASTER_PORT)"
            - "train.py"
            resources:
              limits:
                nvidia.com/gpu: 2
            volumeMounts:
            - name: shm
              mountPath: /dev/shm
          volumes:
          - name: shm
            emptyDir:
              medium: Memory
              sizeLimit: 8Gi
```

### Verification
- [ ] Both master and worker pods Running
- [ ] NCCL initialization successful (check logs)
- [ ] Training completes across both nodes
- [ ] Model saved to /checkpoints/

---

## Lab 4: Ray Serve for Model Inference

### Steps

```bash
# 1. Install KubeRay operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator

# 2. Deploy RayService
kubectl apply -f ray-service.yaml
```

### ray-service.yaml
```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: mnist-serve
spec:
  serveConfigV2: |
    applications:
    - name: mnist
      route_prefix: /predict
      import_path: serve_app:app
      deployments:
      - name: MNISTModel
        num_replicas: 2
        ray_actor_options:
          num_gpus: 0.5
  rayClusterConfig:
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
      template:
        spec:
          containers:
          - name: ray-head
            image: rayproject/ray-ml:2.9.0-py310-gpu
            resources:
              limits:
                cpu: "2"
                memory: "4Gi"
    workerGroupSpecs:
    - replicas: 2
      groupName: gpu-workers
      rayStartParams:
        num-gpus: "1"
      template:
        spec:
          containers:
          - name: ray-worker
            image: rayproject/ray-ml:2.9.0-py310-gpu
            resources:
              limits:
                nvidia.com/gpu: 1
```

### Test Inference
```bash
kubectl port-forward svc/mnist-serve-serve-svc 8000:8000
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"image": [0.1, 0.2, ...]}'
```

### Verification
- [ ] RayService Running
- [ ] Ray dashboard accessible
- [ ] Inference endpoint responds
- [ ] Autoscaling works under load

---

## Lab 5: Configure Karpenter for GPU Auto-Scaling

### Steps

```bash
# 1. Install Karpenter (if not already)
helm install karpenter oci://public.ecr.aws/karpenter/karpenter \
  --namespace karpenter --create-namespace

# 2. Apply NodePool
kubectl apply -f gpu-nodepool.yaml

# 3. Submit GPU job → watch Karpenter provision nodes
kubectl apply -f gpu-job.yaml

# 4. Monitor
kubectl logs -n karpenter -l app.kubernetes.io/name=karpenter -f
# Should see: "launched instance" messages
```

### Verification
- [ ] GPU node provisioned within 2 minutes
- [ ] Correct instance type selected
- [ ] Node removed after pod completes (consolidation)

---

## Lab 6: NCCL All-Reduce Test

### Objective
Verify GPU-to-GPU communication works and measure bandwidth.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: nccl-test
spec:
  template:
    spec:
      containers:
      - name: nccl
        image: nvcr.io/nvidia/pytorch:24.01-py3
        command:
        - bash
        - -c
        - |
          apt-get update && apt-get install -y build-essential
          git clone https://github.com/NVIDIA/nccl-tests.git
          cd nccl-tests
          make MPI=0 CUDA_HOME=/usr/local/cuda
          ./build/all_reduce_perf -b 1M -e 1G -f 2 -g 8
        env:
        - name: NCCL_DEBUG
          value: "INFO"
        resources:
          limits:
            nvidia.com/gpu: 8
        volumeMounts:
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 64Gi
      restartPolicy: Never
```

### Expected Output
```
# 8 GPUs on p4d.24xlarge (A100 + NVLink):
#  size      time(us)  algbw(GB/s)  busbw(GB/s)
#  1M        50        20.0         37.5
#  64M       120       533.3        1000.0
#  1G        1700      617.6        1158.0
```

### Verification
- [ ] NCCL initializes without errors
- [ ] busbw > 500 GB/s for 1GB messages (NVLink)
- [ ] No timeout errors

---

## Lab 7: GPU Failure Simulation + Auto-Recovery

### Objective
Simulate GPU failure and verify checkpointing + recovery works.

### Steps

```bash
# 1. Start training with checkpointing (use Lab 3 script + checkpoint code)
kubectl apply -f training-with-checkpoint.yaml

# 2. Wait for training to progress (check logs for step count)
kubectl logs distributed-training-master-0 -f

# 3. Simulate failure: Kill the worker pod
kubectl delete pod distributed-training-worker-0

# 4. Observe:
# - PyTorchJob controller detects failure
# - New worker pod created
# - Training resumes from last checkpoint

kubectl get pods -w  # Watch pods
kubectl logs distributed-training-master-0 -f  # Watch training logs
# Should see: "Resumed from step XXXX"
```

### Verification
- [ ] Pod deletion detected
- [ ] New pod spawns automatically
- [ ] Training resumes from checkpoint (not step 0!)
- [ ] Final model accuracy same as uninterrupted run

---

## Lab 8: MIG Configuration

### Objective
Partition 1 A100 into 7 MIG instances and run 7 different inference models.

```bash
# On A100 node:

# 1. Enable MIG
sudo nvidia-smi -i 0 -mig 1

# 2. Create 7 x 1g.10gb instances
sudo nvidia-smi mig -i 0 -cgi 19,19,19,19,19,19,19
sudo nvidia-smi mig -i 0 -cci

# 3. Verify
nvidia-smi mig -lgi
# Should show 7 GPU instances

# 4. On K8s: GPU Operator auto-detects MIG
kubectl get nodes -o json | jq '.items[].status.capacity'
# Should show: "nvidia.com/mig-1g.10gb": "7"
```

```yaml
# Deploy 7 inference pods (one per MIG slice)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-model
spec:
  replicas: 7
  template:
    spec:
      containers:
      - name: model
        image: my-inference:latest
        resources:
          limits:
            nvidia.com/mig-1g.10gb: 1   # 1 MIG slice per pod
```

### Verification
- [ ] 7 MIG instances visible in nvidia-smi
- [ ] 7 pods scheduled, each on separate MIG slice
- [ ] No interference between pods (memory isolated)

---

## Lab 9: Volcano Gang Scheduling

### Steps

```bash
# 1. Install Volcano
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml

# 2. Create queue
kubectl apply -f - <<EOF
apiVersion: scheduling.volcano.sh/v1beta1
kind: Queue
metadata:
  name: training
spec:
  weight: 1
  reclaimable: true
  capability:
    nvidia.com/gpu: 8
EOF

# 3. Submit gang-scheduled job
kubectl apply -f volcano-gang-job.yaml
```

### volcano-gang-job.yaml
```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: gang-test
spec:
  minAvailable: 2        # BOTH pods must be scheduled or NONE
  schedulerName: volcano
  queue: training
  tasks:
  - replicas: 2
    name: worker
    template:
      spec:
        schedulerName: volcano
        containers:
        - name: pytorch
          image: nvidia/cuda:12.2.0-base-ubuntu22.04
          command: ["sleep", "60"]
          resources:
            limits:
              nvidia.com/gpu: 1
        restartPolicy: OnFailure
```

### Test Gang Behavior
```bash
# If only 1 GPU available:
# → Both pods should stay Pending (gang = all or nothing)
# Add second GPU node → both pods schedule simultaneously
```

### Verification
- [ ] With insufficient GPUs: both pods Pending
- [ ] With sufficient GPUs: both pods schedule together
- [ ] PodGroup status shows gang behavior

---

## Lab 10: GPU Monitoring Dashboard

### Steps

```bash
# 1. Install Prometheus + Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# 2. DCGM Exporter (already installed with GPU Operator)
# Verify:
kubectl get pods -n gpu-operator | grep dcgm

# 3. Configure Prometheus to scrape DCGM
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: nvidia-dcgm-exporter
  namespaceSelector:
    matchNames:
    - gpu-operator
  endpoints:
  - port: gpu-metrics
    interval: 15s
EOF

# 4. Import Grafana dashboard
# Dashboard ID: 12239 (NVIDIA DCGM Exporter Dashboard)
kubectl port-forward svc/monitoring-grafana 3000:80 -n monitoring
# Open localhost:3000, import dashboard 12239
```

### Verification
- [ ] Prometheus scraping DCGM metrics
- [ ] Grafana shows GPU utilization, memory, temperature
- [ ] Alerts fire when GPU idle > 30 min

---

## Lab 11: Spot Instance Interruption Handling

### Steps

```yaml
# 1. Karpenter NodePool with spot
# (Use from Lab 5 with capacity-type: spot)

# 2. Training job with checkpoint handler
apiVersion: v1
kind: Pod
metadata:
  name: spot-training
spec:
  terminationGracePeriodSeconds: 120  # 2 min to save checkpoint
  containers:
  - name: pytorch
    image: my-training:latest
    command: ["python", "train_with_spot_handler.py"]
    resources:
      limits:
        nvidia.com/gpu: 1
    volumeMounts:
    - name: checkpoints
      mountPath: /checkpoints
  volumes:
  - name: checkpoints
    persistentVolumeClaim:
      claimName: checkpoint-pvc
```

### Simulate Interruption
```bash
# Simulate by draining the node
kubectl drain <node-name> --grace-period=120 --delete-emptydir-data

# Check:
# 1. Pod receives SIGTERM
# 2. Checkpoint saves
# 3. Pod terminates gracefully
# 4. Karpenter launches new node
# 5. Pod reschedules and resumes
```

### Verification
- [ ] Pod handles SIGTERM gracefully
- [ ] Checkpoint saved within grace period
- [ ] Training resumes from checkpoint on new node

---

## Lab 12: Topology-Aware GPU Scheduling

### Steps

```bash
# 1. Check GPU topology
kubectl exec -it gpu-pod -- nvidia-smi topo -m

# 2. Enable topology manager on kubelet
# Edit kubelet config:
# topologyManagerPolicy: "best-effort"
# topologyManagerScope: "pod"

# 3. Deploy pod with topology requirements
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: topo-test
spec:
  containers:
  - name: pytorch
    image: nvidia/cuda:12.2.0-base-ubuntu22.04
    command:
    - bash
    - -c
    - "nvidia-smi topo -m && nvidia-smi -L && sleep 300"
    resources:
      limits:
        nvidia.com/gpu: 4  # Should get 4 NVLink-connected GPUs
  restartPolicy: Never
EOF
```

### Verification
- [ ] nvidia-smi topo shows NVLink between allocated GPUs
- [ ] No SYS (cross-NUMA) connections between allocated GPUs
- [ ] NCCL performance matches expected NVLink bandwidth

---

## Cleanup

```bash
# After all labs:
eksctl delete cluster --name gpu-lab --region us-east-1
# This deletes everything (nodes, EBS, etc.)
# Manually delete: FSx filesystems, S3 buckets, EFS
```

## Summary Checklist

- [ ] Lab 1: GPU Operator deployed
- [ ] Lab 2: KubeFlow running
- [ ] Lab 3: Distributed training works
- [ ] Lab 4: Ray Serve inference running
- [ ] Lab 5: Karpenter auto-scales GPUs
- [ ] Lab 6: NCCL bandwidth verified
- [ ] Lab 7: Fault recovery tested
- [ ] Lab 8: MIG partitioning done
- [ ] Lab 9: Volcano gang scheduling works
- [ ] Lab 10: GPU monitoring dashboard live
- [ ] Lab 11: Spot interruption handled
- [ ] Lab 12: Topology-aware scheduling verified
