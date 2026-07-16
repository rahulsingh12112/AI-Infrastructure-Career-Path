# 04 — SageMaker Training Compiler

## What is Training Compiler?

Compiler that optimizes your training code to run faster on GPUs without code changes.

```
Without Compiler:
├── PyTorch runs in eager mode (one op at a time)
├── Each operation: CPU schedules → GPU executes → result back
├── Overhead per operation (kernel launch, memory allocation)
└── GPU not fully utilized

With Training Compiler:
├── Analyzes your model graph
├── Fuses multiple operations into one GPU kernel
├── Optimizes memory layout
├── Reduces kernel launch overhead
└── 25-50% training speedup (model dependent)
```

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│           Training Compiler Pipeline                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Your PyTorch Code                                       │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────────┐                                       │
│  │ Graph Capture │  ← Traces model execution             │
│  └──────┬───────┘                                       │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │ Graph Optim  │  ← Operator fusion, dead code removal │
│  └──────┬───────┘                                       │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │ Memory Plan  │  ← Optimize tensor memory layout      │
│  └──────┬───────┘                                       │
│         │                                                │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │ Code Gen     │  ← Generate optimized GPU kernels     │
│  └──────┬───────┘                                       │
│         │                                                │
│         ▼                                                │
│  Optimized GPU Code (runs 25-50% faster)                │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Optimizations Performed

```
1. OPERATOR FUSION:
   Before: [MatMul] → [store] → [load] → [Add] → [store] → [load] → [ReLU]
   After:  [MatMul+Add+ReLU] (single kernel, no intermediate storage)
   
2. MEMORY OPTIMIZATION:
   Before: Allocate new tensor for every intermediate result
   After:  Reuse memory (in-place operations where safe)
   
3. LAYOUT OPTIMIZATION:
   Before: Tensor in row-major, but GPU prefers column-major for this op
   After:  Automatically transpose to optimal layout
   
4. KERNEL SCHEDULING:
   Before: Sequential kernel launches with CPU synchronization
   After:  Batch kernel launches, overlap compute with data movement
   
5. MIXED PRECISION HANDLING:
   Before: Manual FP16/BF16 casting
   After:  Automatic precision management with minimal loss
```

## Using Training Compiler on SageMaker

```python
from sagemaker.pytorch import PyTorch
from sagemaker.huggingface import HuggingFace

# Option 1: PyTorch with Training Compiler
estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.p4d.24xlarge",
    framework_version="2.1",
    py_version="py310",
    
    # Enable Training Compiler
    compiler_config={
        "enabled": True,
        "debug": False,          # Set True for verbose compilation logs
    },
    
    hyperparameters={
        "epochs": 10,
        "batch_size": 32,        # Can often increase batch size with compiler!
        "learning_rate": 5e-5,
    },
)

# Option 2: HuggingFace with Training Compiler
hf_estimator = HuggingFace(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.p4d.24xlarge",
    transformers_version="4.28",
    pytorch_version="2.0",
    py_version="py310",
    
    compiler_config={
        "enabled": True,
    },
    
    hyperparameters={
        "model_name": "bert-base-uncased",
        "epochs": 3,
        "batch_size": 64,        # 50% larger batch than without compiler!
    },
)
```

## Training Script Modifications

```python
# Minimal changes needed for Training Compiler:

# 1. Use torch.compile() (PyTorch 2.0+)
import torch

model = MyModel()
model = torch.compile(model, backend="inductor")  # Or SageMaker's backend

# 2. For HuggingFace Transformers:
# No changes needed! SageMaker compiler handles it automatically.

# 3. Important: Avoid dynamic shapes
# BAD (dynamic):
for seq_len in variable_lengths:
    output = model(input[:, :seq_len])  # Different shape each time → recompiles!

# GOOD (pad to fixed length):
output = model(padded_input)  # Same shape always → compile once, run many times
```

## Performance Gains (Benchmarks)

```
Model               │ Without Compiler │ With Compiler │ Speedup
────────────────────┼──────────────────┼───────────────┼─────────
BERT-base           │ 100 samples/sec  │ 140 samples/sec│ 1.4x
GPT-2 medium        │ 50 samples/sec   │ 72 samples/sec │ 1.44x
ResNet-50           │ 800 imgs/sec     │ 1040 imgs/sec │ 1.3x
T5-large            │ 30 samples/sec   │ 45 samples/sec │ 1.5x
ViT-large           │ 200 imgs/sec     │ 270 imgs/sec  │ 1.35x

Key insights:
├── Transformer models benefit most (40-50% speedup)
├── Memory savings allow larger batch sizes
├── First step is slow (compilation) then fast after
├── Not all models benefit equally (simple models = less gain)
└── Dynamic shapes reduce benefit (frequent recompilation)
```

## torch.compile (PyTorch Native — Preferred)

```python
# SageMaker Training Compiler is being succeeded by PyTorch's native torch.compile
# torch.compile is now the recommended approach

import torch

model = MyTransformerModel().cuda()

# Compile with different backends:
# "inductor" — default, good for most models
# "cudagraphs" — best for fixed-size, repetitive workloads
# "aot_eager" — debugging (no real optimization)

compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="max-autotune",      # Spend more time compiling for best perf
    # mode="reduce-overhead", # Minimize framework overhead
    # mode="default",         # Balance compile time vs performance
)

# First forward pass: SLOW (compiling)
# Subsequent passes: FAST (running compiled code)
output = compiled_model(input_data)  # Compiles here
output = compiled_model(input_data)  # Fast from here on
```

## Limitations & When NOT to Use

```
DON'T use when:
├── Dynamic shapes (variable sequence lengths, NLP without padding)
├── Very small models (compilation overhead > training time)
├── Debugging (compiled code hard to debug)
├── Custom CUDA kernels (may not be compatible)
└── Training < 1 hour (compilation takes 5-30 min to amortize)

DO use when:
├── Large transformer models (BERT, GPT, T5, ViT)
├── Fixed input shapes (padded sequences, fixed image size)
├── Long training runs (compilation cost amortized)
├── Memory-bound (compiler reduces memory = bigger batch)
└── Standard PyTorch operations (nn.Linear, nn.Conv2d, etc.)
```

## Interview Questions

1. **Q: Training Compiler se 40% speedup claim hai, but humein sirf 10% mila. Kyun?**
   A: Common reasons: (1) Dynamic shapes → frequent recompilation, (2) Model already memory-efficient (no fusion benefit), (3) Data loading is bottleneck (not compute), (4) Custom ops that compiler can't optimize, (5) Small model (kernel launch overhead not significant). Profile with `torch.profiler` to find actual bottleneck.

2. **Q: torch.compile vs SageMaker Training Compiler — kya fark hai?**
   A: SageMaker Training Compiler was AWS's solution before torch.compile existed. Now PyTorch 2.0+ has native `torch.compile` with `inductor` backend which is industry-standard. Recommendation: Use `torch.compile` directly — it's portable, well-documented, and actively maintained by PyTorch team. SageMaker supports it natively.

3. **Q: Compilation mein 30 min lag rahe hain. Kaise handle karoge?**
   A: (1) Save compiled model graph for reuse (torch._dynamo.reset() + cache), (2) Use `mode="default"` instead of `max-autotune` (faster compile, slightly less perf), (3) For production: compile once, export with torch.export, (4) Amortize over long training (30 min compile for 3 days training = negligible).

4. **Q: Compiler enable kiya toh OOM error aa raha hai. Kyun?**
   A: Compiler trades memory for speed in some optimizations (stores intermediate tensors for fusion). Fix: (1) Reduce batch size slightly, (2) Use `torch.compile(reduce-overhead)` mode, (3) Enable gradient checkpointing alongside compile, (4) Use BF16 mixed precision.
