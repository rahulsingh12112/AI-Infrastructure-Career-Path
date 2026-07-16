# 02 — Distributed Training (Data Parallel & Model Parallel)

## Why Distributed Training?

```
Single GPU:
├── Model fits in 1 GPU (80GB A100)? → Single GPU training ✓
├── Model > 80GB? → Can't fit → NEED model parallelism
├── Training too slow? → NEED data parallelism (more GPUs = faster)

Scale:
├── GPT-3 (175B params) = ~700 GB in FP32 → needs 9+ A100s just for weights!
├── Training LLaMA-70B = weeks on 1 GPU → hours on 512 GPUs
└── Without distribution = impossible to train modern large models
```

## Two Fundamental Strategies

```
┌────────────────────────────────────────────────────────────────┐
│              Distributed Training Strategies                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DATA PARALLEL:                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │          │
│  │Model(全)│  │Model(全)│  │Model(全)│  │Model(全)│          │
│  │Batch 0  │  │Batch 1  │  │Batch 2  │  │Batch 3  │          │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘          │
│       │            │            │            │                  │
│       └──────── AllReduce (sync gradients) ──┘                 │
│       Each GPU has FULL model, different DATA                   │
│                                                                 │
│  MODEL PARALLEL:                                                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │
│  │  GPU 0  │──│  GPU 1  │──│  GPU 2  │──│  GPU 3  │          │
│  │Layer 0-5│  │Layer 6-11│ │Layer12-17│ │Layer18-23│          │
│  │         │  │         │  │         │  │         │           │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘          │
│  Each GPU has PART of model, SAME data flows through           │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Data Parallelism (DDP — Deep Dive)

### How DDP Works

```
Step 1: Copy full model to each GPU
Step 2: Split data batch across GPUs (each gets 1/N of batch)
Step 3: Each GPU does forward pass independently
Step 4: Each GPU computes gradients independently
Step 5: AllReduce — average gradients across all GPUs
Step 6: Each GPU updates weights with averaged gradients
Step 7: All GPUs now have identical model → repeat

Timeline:
GPU 0: [Forward][Backward][AllReduce][Update] → next batch
GPU 1: [Forward][Backward][AllReduce][Update] → next batch
GPU 2: [Forward][Backward][AllReduce][Update] → next batch
GPU 3: [Forward][Backward][AllReduce][Update] → next batch
         ↑ parallel ↑    ↑ sync point ↑
```

### SageMaker DDP (PyTorch DistributedDataParallel)

```python
# SageMaker Estimator with DDP
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_count=4,              # 4 nodes
    instance_type="ml.p4d.24xlarge",  # 8 GPUs each = 32 GPUs total
    framework_version="2.1",
    py_version="py310",
    
    # Enable DDP
    distribution={
        "torch_distributed": {
            "enabled": True,
        }
    },
    
    # Spot training (cost optimization)
    use_spot_instances=True,
    max_wait=86400,
    max_run=72000,
    checkpoint_s3_uri="s3://my-bucket/checkpoints/",
    
    hyperparameters={
        "epochs": 10,
        "batch_size": 64,       # Per GPU batch size
        "learning_rate": 0.001,
    },
)

estimator.fit({"train": "s3://my-bucket/data/train/"})
```

### Training Script (train.py) for DDP

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def train():
    # SageMaker sets these environment variables automatically
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    
    # Model
    model = MyLargeModel().cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # Data — DistributedSampler ensures no data overlap
    dataset = MyDataset(os.environ["SM_CHANNEL_TRAIN"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler, 
                       num_workers=8, pin_memory=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001 * world_size)
    # Note: Scale learning rate with world_size (linear scaling rule)
    
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Important for proper shuffling!
        model.train()
        
        for batch_idx, (data, target) in enumerate(loader):
            data = data.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)
            
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()  # Gradients auto-synced by DDP
            optimizer.step()
            
            if batch_idx % 100 == 0 and rank == 0:
                print(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")
        
        # Save checkpoint (only rank 0)
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(os.environ["SM_MODEL_DIR"], "checkpoint.pt"))
    
    dist.destroy_process_group()

if __name__ == "__main__":
    train()
```

### DDP Scaling Efficiency

```
Ideal: 4 GPUs = 4x speed (linear scaling)
Reality: Communication overhead reduces efficiency

GPUs  │ Speedup │ Efficiency │ Why not linear?
──────┼─────────┼────────────┼──────────────────────
1     │ 1.0x    │ 100%       │ Baseline
2     │ 1.9x   │ 95%        │ Minimal AllReduce overhead
4     │ 3.7x   │ 93%        │ NVLink handles it well
8     │ 7.2x   │ 90%        │ Still mostly NVLink (1 node)
16    │ 13.5x  │ 84%        │ Cross-node communication!
32    │ 25.6x  │ 80%        │ Network becomes bottleneck
64    │ 48.0x  │ 75%        │ More time in AllReduce
256   │ 179.0x │ 70%        │ At scale, still worth it

Key insight: After ~8 GPUs, cross-node communication hurts.
Solution: Faster interconnect (EFA, InfiniBand) + overlap comm with compute
```

## Model Parallelism (Deep Dive)

### Types of Model Parallelism

```
1. PIPELINE PARALLELISM (PP):
   Model split by layers across GPUs
   
   Input → [GPU0: Layers 0-5] → [GPU1: Layers 6-11] → [GPU2: Layers 12-17] → Output
   
   Problem: GPU bubble (only 1 GPU active at a time)
   Solution: Micro-batches (pipeline multiple batches)

2. TENSOR PARALLELISM (TP):
   Single layer split across GPUs
   
   [GPU0: Left half of weight matrix]  → combine
   [GPU1: Right half of weight matrix] → combine
   
   Requires very fast interconnect (NVLink, NOT across nodes)
   Used within a single node

3. EXPERT PARALLELISM (Mixture of Experts):
   Different "expert" sub-networks on different GPUs
   Router sends input to appropriate expert
   
4. SEQUENCE PARALLELISM (SP):
   Long sequences split across GPUs
   Each GPU handles part of sequence length
```

### Pipeline Parallelism — Micro-batching

```
Without micro-batches (naive):
GPU0: [F0][ idle ][ idle ][ idle ][B0]
GPU1: [ idle ][F0][ idle ][ idle ][ idle ][B0]
GPU2: [ idle ][ idle ][F0][ idle ][ idle ][ idle ][B0]
GPU3: [ idle ][ idle ][ idle ][F0][ idle ][ idle ][ idle ][B0]
                                        ← HUGE GPU BUBBLE!

With micro-batches (4 micro-batches):
GPU0: [F0][F1][F2][F3][ B3][B2][B1][B0]
GPU1: [  ][F0][F1][F2][F3][B3][B2][B1][B0]
GPU2: [  ][  ][F0][F1][F2][F3][B3][B2][B1][B0]
GPU3: [  ][  ][  ][F0][F1][F2][F3][B3][B2][B1][B0]
                        ← Much less bubble time!

Bubble ratio = (pp_size - 1) / (num_microbatches + pp_size - 1)
More micro-batches → less bubble → better efficiency
```

### SageMaker Model Parallelism (SMP v2)

```python
# SageMaker Model Parallel with FSDP (Fully Sharded Data Parallel)
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train_mp.py",
    role=role,
    instance_count=4,
    instance_type="ml.p4d.24xlarge",
    framework_version="2.1",
    py_version="py310",
    
    distribution={
        "torch_distributed": {
            "enabled": True,
        },
        "smdistributed": {
            "modelparallel": {
                "enabled": True,
                "parameters": {
                    "tensor_parallel_degree": 4,    # TP within node
                    "pipeline_parallel_degree": 2,   # PP across 2 nodes
                    "expert_parallel_degree": 1,     # No EP
                    "hybrid_shard_degree": 0,        # Auto
                    "sharding_strategy": "HYBRID_SHARD",
                }
            }
        }
    },
    
    hyperparameters={
        "model_name": "meta-llama/Llama-2-70b-hf",
        "batch_size": 4,
        "micro_batches": 8,
    },
)
```

### FSDP (Fully Sharded Data Parallel)

```
Standard DDP:
├── Each GPU has FULL model copy
├── 70B model × 4 GPUs = 280GB total memory (wasteful!)
└── Limited by single GPU memory

FSDP:
├── Model sharded across GPUs (each has 1/N of weights)
├── 70B model ÷ 4 GPUs = ~18GB per GPU (fits!)
├── During computation: gather needed weights → compute → discard
├── Memory efficient but more communication
└── SageMaker's recommended approach for large models

FSDP vs DDP memory:
┌─────────────────────────────────────────────────┐
│ DDP (70B model, 8 GPUs):                        │
│ Each GPU: 70B weights + 70B gradients + 70B opt │
│ = ~630GB per GPU → DOESN'T FIT!                 │
│                                                  │
│ FSDP (70B model, 8 GPUs):                       │
│ Each GPU: 70B/8 weights + gradients sharded     │
│ = ~40GB per GPU → FITS in A100 80GB! ✓          │
└─────────────────────────────────────────────────┘
```

### Training Script with FSDP

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer

def train():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.bfloat16,
    )
    
    # Wrap with FSDP
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,  # Shard within node, replicate across
        mixed_precision=mixed_precision,
        device_id=local_rank,
        auto_wrap_policy=auto_wrap_policy,  # Wrap each transformer layer
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    for epoch in range(3):
        for batch in dataloader:
            input_ids = batch["input_ids"].cuda(local_rank)
            labels = batch["labels"].cuda(local_rank)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Save with FSDP
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        state_dict = model.state_dict()
        if dist.get_rank() == 0:
            torch.save(state_dict, "/opt/ml/model/model.pt")
    
    dist.destroy_process_group()
```

## 3D Parallelism (Combining All)

```
For training very large models (100B+), combine all strategies:

┌─────────────────────────────────────────────────────────────┐
│              3D Parallelism (DP × TP × PP)                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  32 GPUs = 4 nodes × 8 GPUs per node                       │
│                                                              │
│  Tensor Parallel (TP=4): Within node, NVLink                │
│  ┌──────────────────────────────────────────┐               │
│  │ Node 0:                                   │               │
│  │ [GPU0─GPU1─GPU2─GPU3] = TP group 0       │               │
│  │ [GPU4─GPU5─GPU6─GPU7] = TP group 1       │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  Pipeline Parallel (PP=2): Across TP groups                  │
│  TP group 0 (Layers 0-15) → TP group 1 (Layers 16-31)      │
│                                                              │
│  Data Parallel (DP=4): Across nodes                          │
│  Node 0 and Node 1 = same model, different data             │
│  Node 2 and Node 3 = same model, different data             │
│                                                              │
│  Total: TP=4 × PP=2 × DP=4 = 32 GPUs ✓                    │
└─────────────────────────────────────────────────────────────┘

Why this split?
├── TP within node: Needs NVLink speed (intra-node only!)
├── PP across node pairs: Moderate communication
└── DP across all: Only gradient sync (works over network)
```

### SageMaker 3D Parallelism Config

```python
distribution={
    "smdistributed": {
        "modelparallel": {
            "enabled": True,
            "parameters": {
                "tensor_parallel_degree": 4,     # 4 GPUs per TP group
                "pipeline_parallel_degree": 2,    # 2 stages
                # DP auto-calculated: 32 / (4 × 2) = 4
                "microbatches": 8,                # Pipeline efficiency
                "sharding_strategy": "HYBRID_SHARD",
            }
        }
    }
}
```

## Communication Patterns per Strategy

```
Strategy   │ Communication Pattern   │ Volume       │ Required Speed
───────────┼─────────────────────────┼──────────────┼──────────────────
DDP        │ AllReduce (gradients)   │ Model size   │ Medium (network OK)
FSDP       │ AllGather + ReduceScatter│ Model size  │ High
TP         │ AllReduce per layer      │ Activation  │ Very High (NVLink!)
PP         │ Point-to-point (fwd/bwd)│ Activations │ Medium
EP         │ All-to-All              │ Tokens       │ High
```

## Choosing Strategy (Decision Tree)

```
Model fits in 1 GPU?
├── YES → Single GPU (no distribution needed)
│
├── Model fits with FSDP sharding?
│   ├── YES → FSDP (simplest distributed approach)
│   │   └── Scale out with more DP replicas for speed
│   └── NO → Need Pipeline + Tensor Parallelism
│
└── NO → 3D Parallelism:
    ├── TP degree = min(GPUs_per_node, model_hidden/1024)
    ├── PP degree = num_layers / layers_that_fit_per_TP_group
    └── DP degree = total_GPUs / (TP × PP)

Rules of thumb:
├── TP ≤ 8 (always within one node, NVLink required)
├── PP = 2-8 (more PP = more pipeline bubble)
├── DP = as many as possible (scales best)
├── Micro-batches ≥ 4 × PP (reduce bubble)
└── Effective batch size = micro_batch × DP × accumulation_steps
```

## Interview Questions

1. **Q: 70B model train karna hai 64 GPUs pe (8 nodes × 8 GPUs). Strategy kya hogi?**
   A: FSDP with HYBRID_SHARD. Shard model within each node (8-way), replicate across nodes (8 DP replicas). This gives: ~9GB model per GPU + gradients + optimizer ≈ 45GB per GPU (fits A100 80GB). Alternative: TP=4, PP=2, DP=8 for even larger batch sizes.

2. **Q: Data Parallel training mein 64 GPU pe accuracy drop ho rahi hai. Kyun?**
   A: Effective batch size too large (64 × per-GPU batch). Large batch = less noise = worse generalization. Fix: (1) Linear learning rate scaling with warmup, (2) Gradient accumulation to reduce effective batch, (3) LARS/LAMB optimizer designed for large batch, (4) Reduce per-GPU batch and increase accumulation steps.

3. **Q: Tensor Parallelism cross-node kyun nahi karte?**
   A: TP requires AllReduce EVERY layer (very frequent, small messages). Cross-node latency (microseconds) × num_layers × steps = massive overhead. NVLink (600 GB/s) vs network (50 GB/s) = 12x speed difference. TP only works with NVLink-speed connections (intra-node).

4. **Q: Pipeline parallelism mein bubble fraction 40% hai. Kaise reduce karoge?**
   A: (1) Increase micro-batches (bubble = (PP-1)/(micro_batches+PP-1)), (2) Use interleaved pipeline schedule (1F1B), (3) Reduce PP degree (try TP=8, PP=1 within node), (4) Async pipeline (don't wait for full backward), (5) Virtual stages (more stages than physical GPUs).
