# 08 — Storage for ML Workloads

## Why Storage Matters

```
GPU can process: 2 TB/s (HBM bandwidth)
Network to GPU:  50 GB/s (InfiniBand)
Storage to CPU:  3-12 GB/s (NVMe SSD)
S3 download:     0.5-2 GB/s

Gap: GPU is 1000x faster than storage
→ If data pipeline is slow, GPU sits idle waiting for data
→ This is the #1 performance bottleneck in practice!
```

## Data Flow in ML Training

```
┌──────────────────────────────────────────────────────────┐
│                    Data Pipeline                           │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  S3 (cold)                                                │
│  └── Download → FSx Lustre (warm cache)                  │
│       └── DataLoader prefetch → CPU RAM (hot)            │
│            └── Pin memory → GPU Memory (active)          │
│                 └── Training computation                  │
│                                                           │
│  Each stage needs to be fast enough that GPU never waits │
│                                                           │
│  Bandwidth at each stage:                                │
│  S3 → FSx:        1-10 GB/s (bulk transfer, one-time)   │
│  FSx → CPU:       10-50 GB/s (parallel filesystem)      │
│  CPU → GPU:       32 GB/s (PCIe Gen4)                    │
│  GPU HBM:         2 TB/s (internal)                      │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## Storage Options Comparison

```
Storage         │ Throughput  │ Latency │ Cost      │ Access    │ Best For
────────────────┼─────────────┼─────────┼───────────┼───────────┼─────────────
Local NVMe      │ 7 GB/s      │ <1ms    │ Included  │ Single    │ Cache, temp
(instance store)│             │         │           │ node      │

EBS gp3         │ 1 GB/s      │ 1-5ms   │ $0.08/GB  │ Single    │ OS, small
                │             │         │           │ node      │ datasets

EBS io2         │ 4 GB/s      │ <1ms    │ $0.125/GB │ Single    │ High-perf
                │             │         │           │ node      │ single-node

EFS             │ 3-10 GB/s   │ 1-5ms   │ $0.30/GB  │ Multi     │ Shared config
(NFS)           │             │         │           │ node      │ notebooks

FSx Lustre      │ 100+ GB/s   │ <1ms    │ $0.14/GB  │ Multi     │ Training data
                │ (cluster)   │         │           │ node      │ checkpoints

S3              │ 100+ GB/s   │ 10-50ms │ $0.023/GB │ Multi     │ Cold data
                │ (aggregate) │         │           │ node      │ archival

S3 + Mountpoint │ 10-50 GB/s  │ 10-50ms │ $0.023/GB │ Multi     │ Read-heavy
                │             │         │           │ node      │ datasets
```

## FSx for Lustre (Primary Choice for ML)

### What is Lustre?
Parallel filesystem — multiple storage servers serve data simultaneously.
Like reading from 100 books at the same time instead of 1.

```
Architecture:
┌─────────────────────────────────────────────────────┐
│               FSx for Lustre Cluster                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  MDT (Metadata)     OSTs (Object Storage Targets)   │
│  ┌──────────┐      ┌─────┐ ┌─────┐ ┌─────┐        │
│  │ File names│      │OST 0│ │OST 1│ │OST 2│ ...    │
│  │ Directory │      │     │ │     │ │     │        │
│  │ structure │      │Data │ │Data │ │Data │        │
│  └──────────┘      │chunk│ │chunk│ │chunk│        │
│                     └─────┘ └─────┘ └─────┘        │
│                                                      │
│  File "model.bin" (100 GB):                         │
│  ├── Chunk 1 (1MB) → OST 0                         │
│  ├── Chunk 2 (1MB) → OST 1                         │
│  ├── Chunk 3 (1MB) → OST 2                         │
│  └── ... striped across all OSTs                    │
│                                                      │
│  Parallel read: All OSTs serve simultaneously       │
│  = N × single-disk speed                            │
└─────────────────────────────────────────────────────┘
```

### FSx Configuration for ML

```yaml
# FSx for Lustre (Terraform/CloudFormation style)
Type: PERSISTENT_2
StorageCapacity: 4800      # GB (multiples of 2400 for PERSISTENT_2)
DeploymentType: PERSISTENT_2
PerUnitStorageThroughput: 250  # MB/s per TiB (125, 250, 500, 1000)

# Calculated throughput:
# 4800 GB = 4.7 TiB × 250 MB/s/TiB = ~1.2 GB/s baseline
# Burst: Up to 6x for reads from S3-linked data

# S3 Data Repository Association:
DataRepositoryAssociations:
  - FilesystemPath: /training-data
    DataRepositoryPath: s3://my-training-bucket/
    AutoImportPolicy: NEW_CHANGED_DELETED
    # FSx automatically imports new S3 objects!
```

### FSx on Kubernetes (CSI Driver)

```yaml
# StorageClass for FSx Lustre
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fsx-lustre-sc
provisioner: fsx.csi.aws.com
parameters:
  subnetId: subnet-xxxxx
  securityGroupIds: sg-xxxxx
  deploymentType: PERSISTENT_2
  perUnitStorageThroughput: "250"
  dataRepositoryAssociations: '[{"fileSystemPath":"/data","dataRepositoryPath":"s3://my-bucket/"}]'
mountOptions:
  - flock

---
# PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: fsx-lustre-sc
  resources:
    requests:
      storage: 4800Gi

---
# Pod using FSx
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: training
    image: my-training:latest
    volumeMounts:
    - name: data
      mountPath: /data
    - name: checkpoints
      mountPath: /checkpoints
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: training-data
```

## S3 Mountpoint for Amazon S3

```yaml
# S3 Mountpoint — mount S3 bucket as filesystem
# Good for: Read-heavy, large datasets, cost-effective
# Bad for: Random writes, small files

apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: s3-mountpoint-sc
provisioner: s3.csi.aws.com
parameters:
  bucketName: my-training-data

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: s3-data
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: s3-mountpoint-sc
  resources:
    requests:
      storage: 1Ti   # Not actually enforced for S3
```

## Data Loading Optimization

### PyTorch DataLoader Best Practices

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=8,          # Parallel data loading (CPU cores)
    pin_memory=True,        # Pin to page-locked memory (faster GPU transfer)
    prefetch_factor=4,      # Prefetch 4 batches ahead
    persistent_workers=True, # Don't kill workers between epochs
    drop_last=True,         # Avoid small last batch
)

# Rule of thumb:
# num_workers = number of CPU cores / number of GPUs on node
# 96 cores, 8 GPUs → num_workers = 12 per DataLoader
```

### Data Format Matters

```
Format      │ Read Speed │ Random Access │ Best For
────────────┼────────────┼──────────────┼──────────────────
Raw images  │ Slow       │ Yes          │ Nothing (avoid!)
TFRecord    │ Fast       │ No           │ TensorFlow
WebDataset  │ Very fast  │ No           │ Large-scale training
Parquet     │ Fast       │ Columnar     │ Tabular data
HDF5        │ Medium     │ Yes          │ Scientific data
MosaicML    │ Very fast  │ Yes          │ LLM pre-training
Streaming   │

Best practice for large-scale:
├── Convert raw data → WebDataset/MosaicML streaming format
├── Store as large sequential shards (256MB-1GB each)
├── Each worker reads different shards (no contention)
└── Sequential reads = maximum throughput from FSx/S3
```

### Caching Strategy

```
Multi-level cache:

Level 1: GPU memory (torch.cuda cache)
├── Reused tensors stay in GPU memory
├── torch.cuda.empty_cache() only when OOM

Level 2: CPU pinned memory (DataLoader prefetch)
├── pin_memory=True
├── Prefetch next N batches while GPU computes current

Level 3: Local NVMe cache (instance store)
├── First epoch: Read from FSx, cache to NVMe
├── Subsequent epochs: Read from NVMe (7 GB/s vs 1 GB/s)
├── Works great for datasets that fit on local disk

Level 4: FSx Lustre (shared filesystem)
├── Lazy loading from S3: First access fetches from S3
├── Subsequent access: Served from FSx cache
├── Transparent to application

Level 5: S3 (cold storage)
├── Source of truth
├── Accessed only on cache miss
```

## Storage Architecture for Different Workloads

### Small Model Training (<10B params)
```
Architecture:
S3 → EFS (shared) → DataLoader → GPU

Why EFS:
├── Simple setup
├── Shared across nodes
├── Cheaper than FSx for small data
├── Sufficient throughput for small models
```

### Large Model Training (10B-100B+ params)
```
Architecture:
S3 → FSx Lustre → Local NVMe cache → DataLoader → GPU

Why FSx:
├── Parallel reads = 100+ GB/s aggregate
├── Large checkpoint writes are fast
├── S3 auto-sync (data appears automatically)
├── Multiple nodes can read same data simultaneously

Checkpoints: FSx (fast writes, shared access)
Training data: FSx + S3 lazy load
```

### Inference / Model Serving
```
Architecture:
S3 → Local NVMe (model weights) → GPU memory

Why Local NVMe:
├── Model loaded once at startup
├── Need fast cold-start (7 GB/s from NVMe)
├── No shared access needed
├── Cheapest for read-once pattern
```

## Troubleshooting Storage Issues

```bash
# Check FSx performance
lctl get_param llite.*.stats    # Lustre client stats
lfs df -h                        # Lustre disk free
lfs getstripe /data/file         # Check file striping

# Check I/O bottleneck
iostat -xz 1                     # Disk I/O stats
dstat --disk --io                # Real-time I/O monitoring

# Check if GPU is waiting for data
nvidia-smi dmon -s u -d 1        # If GPU util fluctuates = data bottleneck
# Pattern: 95%...20%...95%...20% = DataLoader can't keep up

# Check DataLoader performance
# In PyTorch: Add timing around data loading
import time
for batch in dataloader:
    data_time = time.time()  # If this is >50% of iteration = bottleneck
    output = model(batch)
    compute_time = time.time()
```

## Interview Questions

1. **Q: 1TB dataset, 32 GPU training job. Storage architecture kya hogi?**
   A: S3 (source) → FSx Lustre 4.8TB PERSISTENT_2 with S3 association → auto-import on first read → DataLoader with num_workers=12, pin_memory=True → Local NVMe for caching (if dataset < instance storage). Checkpoints also on FSx (shared, fast writes).

2. **Q: Training mein GPU utilization 40% hai, memory 90%. Problem kya hai?**
   A: GPU memory full hai (model fits) but compute utilization low = data loading slow. Fix: increase num_workers, use prefetch_factor, check storage throughput (lfs stats), consider local NVMe cache, convert to streaming format (WebDataset).

3. **Q: FSx vs EFS — kab kya use karna?**
   A: FSx: Large-scale training (parallel reads, high throughput, S3 integration). EFS: Shared notebooks, configs, small datasets, simple setup. FSx costs more but 10-50x faster for large sequential reads.

4. **Q: Checkpoint 200GB hai, FSx pe save ho raha hai 2 min mein. Tolerable hai?**
   A: Depends. If training step = 30 sec, then 2 min save = blocking 4 steps = bad. Solution: Async checkpoint (save in background), or distributed checkpoint (each rank saves 25GB in parallel = 15 sec), or save to local NVMe first then async copy to FSx.
