# 07 — Checkpointing & Fault Tolerance

## The Problem

```
Training a large model:
├── Takes 3-14 days on 32+ GPUs
├── Cost: $10,000 - $100,000+
├── If 1 GPU fails on day 6 → ALL progress lost!
├── GPU failure rate at scale: ~1-5% per day per 1000 GPUs
└── Without checkpointing = gambling with $$$
```

## Checkpointing Basics

```
What is checkpointing?
├── Periodically save model state to storage
├── State includes: model weights, optimizer state, learning rate, epoch/step
├── If failure → load last checkpoint → continue from there
└── Like "save game" in video games

Timeline:
├──Step 0──────Step 1000──────Step 2000──────Step 3000──────✗ CRASH
│              [Checkpoint 1]  [Checkpoint 2]  [Checkpoint 3]
│
│  After crash: Resume from Checkpoint 3 (Step 3000)
│  Lost: Only steps 3000-3xxx (minutes, not days)
```

## PyTorch Checkpointing

```python
import torch
import os

# ============ SAVE CHECKPOINT ============
def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'global_step': step,
        'loss': loss,
        'rng_state': torch.random.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
    }
    # Save to temp file first (atomic write)
    temp_path = path + '.tmp'
    torch.save(checkpoint, temp_path)
    os.rename(temp_path, path)  # Atomic rename
    print(f"Checkpoint saved at step {step}")

# ============ LOAD CHECKPOINT ============
def load_checkpoint(model, optimizer, scheduler, path):
    if not os.path.exists(path):
        return 0, 0  # No checkpoint, start fresh
    
    checkpoint = torch.load(path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    torch.random.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    
    print(f"Resumed from step {checkpoint['global_step']}")
    return checkpoint['epoch'], checkpoint['global_step']

# ============ TRAINING LOOP ============
# Resume if checkpoint exists
start_epoch, start_step = load_checkpoint(model, optimizer, scheduler, 
                                           '/shared-storage/checkpoints/latest.pt')

for epoch in range(start_epoch, total_epochs):
    for step, batch in enumerate(dataloader, start=start_step):
        loss = train_step(model, batch)
        
        # Save every 1000 steps
        if step % 1000 == 0:
            save_checkpoint(model, optimizer, scheduler, 
                          epoch, step, loss,
                          '/shared-storage/checkpoints/latest.pt')
```

## Distributed Checkpointing (Multi-GPU)

```python
import torch.distributed as dist
from torch.distributed.checkpoint import save, load

# ============ DISTRIBUTED SAVE (FSDP/DDP) ============
# Each rank saves its shard — much faster than gathering to rank 0

# Option 1: Only rank 0 saves (simple but slow for large models)
if dist.get_rank() == 0:
    torch.save(model.state_dict(), path)
dist.barrier()  # All ranks wait

# Option 2: Distributed checkpoint (fast, each rank saves its piece)
from torch.distributed.checkpoint import save_state_dict, load_state_dict
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader

writer = FileSystemWriter("/shared-storage/checkpoints/step-5000/")
save_state_dict(
    state_dict={"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    storage_writer=writer,
)

# Load (each rank loads its shard)
reader = FileSystemReader("/shared-storage/checkpoints/step-5000/")
load_state_dict(
    state_dict={"model": model.state_dict(), "optimizer": optimizer.state_dict()},
    storage_reader=reader,
)
```

## Async Checkpointing (Don't Block Training!)

```python
import threading
import torch

class AsyncCheckpointer:
    """Save checkpoints in background thread — training doesn't stop"""
    
    def __init__(self):
        self._thread = None
    
    def save(self, state_dict, path):
        # Wait for previous save to finish
        if self._thread is not None:
            self._thread.join()
        
        # Copy state to CPU (so GPU can continue training)
        cpu_state = {k: v.cpu().clone() for k, v in state_dict.items()}
        
        # Save in background
        self._thread = threading.Thread(
            target=self._save_worker, 
            args=(cpu_state, path)
        )
        self._thread.start()
    
    def _save_worker(self, state_dict, path):
        temp_path = path + '.tmp'
        torch.save(state_dict, temp_path)
        os.rename(temp_path, path)

# Usage:
checkpointer = AsyncCheckpointer()
# This returns immediately — training continues while saving happens
checkpointer.save(model.state_dict(), '/checkpoints/step-5000.pt')
```

## Checkpointing on Kubernetes

### Storage Options for Checkpoints

```
Storage           │ Speed    │ Cost   │ Best For
──────────────────┼──────────┼────────┼──────────────────
Local NVMe        │ Fastest  │ Lost on│ Temp cache only
                  │          │ evict  │
EFS (NFS)         │ Medium   │ Low    │ Small models (<10GB)
FSx for Lustre    │ Fast     │ Medium │ Large models, frequent ckpt
S3                │ Slow     │ Cheapest│ Final checkpoints, archival
S3 + Mountpoint   │ Medium   │ Low    │ Good balance
```

### K8s Config for Checkpointing

```yaml
# PVC for checkpoint storage (EFS example)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: checkpoint-storage
spec:
  accessModes:
    - ReadWriteMany        # Multiple pods can write (distributed ckpt)
  storageClassName: efs-sc
  resources:
    requests:
      storage: 500Gi

---
# Training pod with checkpoint volume
apiVersion: v1
kind: Pod
metadata:
  name: training-worker-0
spec:
  containers:
  - name: pytorch
    image: my-training:latest
    env:
    - name: CHECKPOINT_DIR
      value: "/checkpoints"
    - name: CHECKPOINT_INTERVAL
      value: "1000"          # Save every 1000 steps
    volumeMounts:
    - name: checkpoints
      mountPath: /checkpoints
    - name: shm
      mountPath: /dev/shm
    resources:
      limits:
        nvidia.com/gpu: 8
  volumes:
  - name: checkpoints
    persistentVolumeClaim:
      claimName: checkpoint-storage
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi
```

## TorchElastic (torchrun) — Auto-Recovery

```bash
# torchrun = PyTorch's built-in fault tolerance
# If a worker dies → automatically restarts all workers → resumes from checkpoint

torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --max_restarts=3 \          # Auto-restart up to 3 times
  --rdzv_backend=c10d \       # Rendezvous backend
  --rdzv_endpoint=master:29400 \
  train.py \
  --checkpoint_dir /checkpoints
```

### TorchElastic on K8s (Elastic Training)

```yaml
# PyTorchJob with elastic training
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: elastic-training
spec:
  elasticPolicy:
    minReplicas: 2          # Min workers to continue
    maxReplicas: 4          # Max workers
    maxRestarts: 5          # Auto-restart on failure
    rdzvBackend: c10d
  pytorchReplicaSpecs:
    Worker:
      replicas: 4
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: my-training:latest
            command:
            - torchrun
            - --nnodes=2:4    # Elastic: 2 to 4 nodes
            - --nproc_per_node=8
            - --max_restarts=5
            - train.py
            resources:
              limits:
                nvidia.com/gpu: 8
```

## Tiered Checkpointing Strategy

```
Tier 1: In-Memory (every 100 steps)
├── Save to GPU memory / pinned CPU memory
├── Speed: Instant
├── Survives: Software crash
├── Lost on: Node failure

Tier 2: Local NVMe (every 500 steps)
├── Save to local SSD
├── Speed: Very fast (5-10 GB/s)
├── Survives: Process crash, GPU reset
├── Lost on: Node replacement

Tier 3: Shared Storage (every 1000 steps)
├── Save to FSx/EFS/S3
├── Speed: Medium (1-3 GB/s)
├── Survives: Node failure, cluster issues
├── Never lost (durable storage)

Strategy:
├── Most failures = software → Tier 1 handles (fast recovery)
├── GPU failures = Tier 2 handles (local NVMe survives)
├── Node failures = Tier 3 handles (shared storage survives)
└── Only lose work since last tier-appropriate checkpoint
```

## SageMaker HyperPod Auto-Recovery

```
HyperPod handles fault tolerance automatically:

1. Health monitoring agent on every node
2. Detects: GPU failure, node unreachable, ECC errors
3. Automatic actions:
   ├── Cordon failed node
   ├── Launch replacement node
   ├── Resume training from last checkpoint
   └── No human intervention needed

Your responsibility:
├── Write proper checkpointing in training code
├── Save to shared storage (S3/FSx)
└── Use torchrun for elastic training

HyperPod handles:
├── Hardware detection
├── Node replacement
├── Cluster health monitoring
└── Auto-resume trigger
```

## Checkpoint Size & Frequency Optimization

```
Model Size    │ Checkpoint Size  │ Save Time (FSx) │ Recommended Interval
──────────────┼──────────────────┼─────────────────┼──────────────────
1B params     │ ~4 GB (FP32)     │ ~2 sec          │ Every 500 steps
7B params     │ ~28 GB           │ ~15 sec         │ Every 1000 steps
70B params    │ ~280 GB          │ ~2 min          │ Every 2000 steps
175B params   │ ~700 GB          │ ~5 min          │ Every 3000 steps

Optimization:
├── Use FP16/BF16 checkpoints (half the size)
├── Only save model + optimizer (skip gradients)
├── Distributed checkpoint (parallel writes)
├── Async save (don't block training)
├── Keep only last N checkpoints (rolling)
└── Compress if storage is bottleneck
```

## Rolling Checkpoint Strategy

```python
import glob
import os

MAX_CHECKPOINTS = 3  # Keep only last 3

def save_rolling_checkpoint(state, step, checkpoint_dir):
    path = os.path.join(checkpoint_dir, f'checkpoint-{step}.pt')
    torch.save(state, path)
    
    # Delete old checkpoints (keep last 3)
    all_ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*.pt')))
    while len(all_ckpts) > MAX_CHECKPOINTS:
        os.remove(all_ckpts.pop(0))  # Remove oldest
    
    # Update symlink to latest
    latest_link = os.path.join(checkpoint_dir, 'latest.pt')
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    os.symlink(path, latest_link)
```

## Interview Questions

1. **Q: 70B model train kar rahe ho 32 GPUs pe. Checkpoint strategy kya hogi?**
   A: Distributed checkpoint (each rank saves its shard in parallel) → shared FSx storage → every 2000 steps → async save → keep last 3 → total ~140GB per checkpoint (BF16). Recovery time: ~2 min to load. Training interruption: <10 sec (async).

2. **Q: Checkpoint save mein 5 min lag rahe hain. Training slow ho raha hai. Fix?**
   A: (1) Async checkpointing — save in background thread, (2) Distributed checkpoint — parallel writes, (3) Save to local NVMe first, then async copy to shared storage, (4) Use BF16 instead of FP32 (half size), (5) Reduce frequency if loss is stable.

3. **Q: Node fail hua mid-training. Kaise resume hoga automatically?**
   A: (1) torchrun detects worker failure → waits for replacement, (2) New node joins → all workers restart, (3) Each worker loads last checkpoint from shared storage, (4) DataLoader resumes from correct position, (5) Training continues. Total downtime: ~5-10 min (node replacement + checkpoint load).

4. **Q: Checkpoint file corrupt ho gaya. Kaise handle karoge?**
   A: (1) Rolling checkpoints — try previous checkpoint (step-2000 instead of step-3000), (2) Atomic writes — save to .tmp first, then rename (prevents partial writes), (3) Checksum verification after save, (4) Keep minimum 3 checkpoints always.
