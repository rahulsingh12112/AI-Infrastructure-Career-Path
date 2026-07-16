# 04 — Quantization: GPTQ, AWQ, INT4/INT8

## What is Quantization?

Reduce model precision (fewer bits per weight) to make it smaller and faster.

```
Full precision (FP32):  32 bits per weight → 7B model = 28 GB
Half precision (FP16):  16 bits per weight → 7B model = 14 GB
INT8 quantization:       8 bits per weight → 7B model = 7 GB
INT4 quantization:       4 bits per weight → 7B model = 3.5 GB

Result:
├── 4-8x smaller model → fits on cheaper/smaller GPU
├── 2-4x faster inference → less memory bandwidth needed
├── Minimal accuracy loss (0.1-1% for good methods)
└── 70B model on single A100? Yes, with INT4!
```

## Precision Formats

```
Format  │ Bits │ Range              │ Use Case
────────┼──────┼────────────────────┼──────────────────────────
FP32    │ 32   │ ±3.4 × 10^38      │ Training (full precision)
BF16    │ 16   │ ±3.4 × 10^38      │ Training + inference (same range as FP32)
FP16    │ 16   │ ±65504             │ Inference (good precision)
FP8     │ 8    │ ±448 (E4M3)       │ H100 native, training+inference
INT8    │ 8    │ -128 to 127        │ Inference (good)
INT4    │ 4    │ -8 to 7            │ Inference (aggressive, some quality loss)
NF4     │ 4    │ Normal distribution │ QLoRA, BitsAndBytes (optimized for NN weights)

Memory per 7B model:
FP32:  28.0 GB
BF16:  14.0 GB
FP16:  14.0 GB
INT8:   7.0 GB
INT4:   3.5 GB
```

## Quantization Methods Comparison

```
Method       │ Bits │ Quality │ Speed    │ Calibration │ Best For
─────────────┼──────┼─────────┼──────────┼─────────────┼──────────────
GPTQ         │ 4/8  │ ⭐⭐⭐⭐  │ ⭐⭐⭐⭐   │ Required    │ Best quality INT4
AWQ          │ 4    │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐⭐  │ Required    │ Best overall
BitsAndBytes │ 4/8  │ ⭐⭐⭐   │ ⭐⭐⭐    │ Not needed  │ Quick & easy
GGUF/GGML    │ 2-8  │ ⭐⭐⭐   │ ⭐⭐⭐    │ Not needed  │ CPU inference
FP8          │ 8    │ ⭐⭐⭐⭐⭐ │ ⭐⭐⭐⭐⭐  │ Not needed  │ H100 native
SmoothQuant  │ 8    │ ⭐⭐⭐⭐  │ ⭐⭐⭐⭐   │ Required    │ INT8 W8A8
```

## GPTQ (Post-Training Quantization)

### How GPTQ Works
```
GPTQ = Optimal Brain Quantization for GPT models

Process:
1. Load full-precision model
2. Feed calibration data (128-1024 samples)
3. For each layer:
   a. Quantize weights column-by-column
   b. Minimize output error: ||W·X - Q(W)·X||²
   c. Adjust remaining columns to compensate for quantization error
   d. Use Hessian (second-order info) for optimal adjustment
4. Save quantized model

Key insight: Don't just round weights — compensate for errors!
Standard rounding: Each weight independently rounded → errors accumulate
GPTQ: When weight A is rounded, adjust B,C,D to compensate → total error minimized
```

### Using GPTQ

```python
# Quantize a model with GPTQ
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Quantization config
quantization_config = GPTQConfig(
    bits=4,                      # 4-bit quantization
    group_size=128,              # Quantize in groups of 128 weights
    dataset="wikitext2",         # Calibration dataset
    tokenizer=tokenizer,
    desc_act=True,               # Order by activation magnitude (better quality)
    sym=False,                   # Asymmetric quantization
)

# Load and quantize
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

# Save quantized model
model.save_pretrained("./llama-7b-gptq-4bit")
tokenizer.save_pretrained("./llama-7b-gptq-4bit")

# Load quantized model for inference
model = AutoModelForCausalLM.from_pretrained(
    "./llama-7b-gptq-4bit",
    device_map="auto",
)
```

### GPTQ Parameters Explained
```
bits: 4 or 8
├── 4-bit: Maximum compression (7B = 3.5GB)
└── 8-bit: Better quality, less compression (7B = 7GB)

group_size: 32, 64, 128, 256
├── Smaller = better quality (more granular scaling factors)
├── Larger = more compression (fewer scaling factors stored)
├── 128 = good default balance
└── -1 = per-channel (no groups, best quality, least compression)

desc_act (descending activation):
├── True: Process columns by activation magnitude (important ones first)
├── Better quality but slower quantization
└── Recommended: True for best accuracy

sym (symmetric):
├── True: Range [-max, max] (same scale for positive/negative)
├── False: Range [min, max] (asymmetric, more precise)
└── False generally better for LLMs
```

## AWQ (Activation-Aware Weight Quantization)

### How AWQ Works
```
AWQ = Protect important weights from quantization damage

Key insight: Not all weights are equally important!
├── 1% of weights are "salient" (critical for accuracy)
├── These correspond to large activation magnitudes
├── Quantizing these = big accuracy loss
├── Protecting these = maintain accuracy with aggressive quantization

Method:
1. Run calibration data through model
2. Identify which weights produce large activations (salient weights)
3. Scale salient channels UP before quantization
4. Scale corresponding activations DOWN (mathematically equivalent)
5. Now salient weights have more quantization precision
6. Result: Same 4-bit, but much better accuracy!

Why it's faster than GPTQ:
├── GPTQ: Solves optimization per column (slow, hours for 70B)
├── AWQ: Just identifies + scales salient weights (fast, minutes)
└── Both achieve similar quality, AWQ is 10x faster to quantize
```

### Using AWQ

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "./llama-7b-awq"

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Quantize
quant_config = {
    "zero_point": True,     # Asymmetric quantization
    "q_group_size": 128,    # Group size
    "w_bit": 4,             # 4-bit weights
    "version": "GEMM",      # Optimized kernel type
}

model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# Inference with vLLM (fastest)
# python -m vllm.entrypoints.openai.api_server \
#   --model ./llama-7b-awq --quantization awq
```

## INT8 Quantization (SmoothQuant / LLM.int8())

### BitsAndBytes (Easy, No Calibration)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit inference (no calibration needed!)
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)

# 4-bit inference (NF4 format, used with QLoRA)
model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # Normal Float 4 (optimized)
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,       # Quantize the quantization constants
    ),
    device_map="auto",
)
```

## FP8 (H100 Native)

```python
# FP8 — native on H100/H200 GPUs
# Best quality-to-compression ratio
# No calibration needed for E4M3 format

# With vLLM:
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Llama-2-7b-hf \
#   --quantization fp8

# With TensorRT-LLM:
# Automatically converts to FP8 during engine build
# Near-zero accuracy loss, 2x throughput improvement on H100
```

## Quality vs Compression Trade-off

```
Model: LLaMA-2-7B on common benchmarks

Quantization    │ Size  │ Perplexity │ MMLU  │ Speed vs FP16
────────────────┼───────┼────────────┼───────┼──────────────
FP16 (baseline) │ 14 GB │ 5.47       │ 46.0% │ 1.0x
FP8             │ 7 GB  │ 5.48       │ 45.9% │ 1.8x
INT8 (SQ)       │ 7 GB  │ 5.50       │ 45.7% │ 1.6x
AWQ-4bit        │ 3.5 GB│ 5.60       │ 45.3% │ 2.2x
GPTQ-4bit       │ 3.5 GB│ 5.62       │ 45.1% │ 2.0x
BnB-4bit (NF4)  │ 3.5 GB│ 5.78       │ 44.5% │ 1.5x
INT4 (naive)    │ 3.5 GB│ 6.80       │ 40.2% │ 2.0x

Key takeaways:
├── AWQ 4-bit: Best quality at 4-bit, recommended
├── GPTQ 4-bit: Close to AWQ, more established
├── FP8: Best if you have H100 (near-zero loss)
├── BnB: Easiest (no calibration) but slightly worse quality
└── Naive INT4: Don't use! (significant quality loss)
```

## When to Use Which

```
Decision Matrix:

Have H100/H200?
└── YES → Use FP8 (best quality, native support, easy)

Need maximum compression (single small GPU)?
└── YES → AWQ 4-bit (best quality at 4-bit)

Quick experiment, no calibration data?
└── YES → BitsAndBytes 4-bit (instant, decent quality)

Production deployment, maximum throughput?
└── AWQ 4-bit with vLLM (best speed + quality combo)

Fine-tuning quantized model (QLoRA)?
└── BitsAndBytes NF4 (designed for QLoRA)

CPU inference (no GPU)?
└── GGUF/GGML format with llama.cpp

TensorRT-LLM deployment?
└── FP8 or INT4 with TRT-LLM's own quantization
```

## Quantization on Kubernetes (Production)

```yaml
# Deploy AWQ-quantized LLaMA-70B on single A100
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-70b-awq
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "TheBloke/Llama-2-70B-Chat-AWQ"
        - "--quantization"
        - "awq"
        - "--tensor-parallel-size"
        - "1"                    # 70B AWQ = ~35GB, fits on 1× A100 80GB!
        - "--gpu-memory-utilization"
        - "0.95"
        - "--max-model-len"
        - "4096"
        resources:
          limits:
            nvidia.com/gpu: 1    # Just 1 GPU for 70B model!
```

## Interview Questions

1. **Q: 70B model deploy karna hai single A100 (80GB) pe. Possible hai?**
   A: Yes with quantization! 70B in FP16 = 140GB (doesn't fit). 70B in AWQ 4-bit = ~35GB (fits!). Use: vLLM with --quantization awq. Leaves ~40GB for KV-cache = can handle ~50-80 concurrent requests. Quality loss: <1% on most benchmarks.

2. **Q: AWQ vs GPTQ — kya fark hai technically?**
   A: Both are 4-bit post-training quantization. GPTQ: Column-by-column weight optimization using Hessian (mathematically optimal, slow to quantize). AWQ: Identifies salient weights by activation magnitude, scales them to protect from quantization (faster to quantize, slightly better quality at inference because it targets what matters most). In practice: AWQ slightly better quality + faster quantization + better kernel support.

3. **Q: Quantized model ki accuracy drop ho rahi hai production mein. Kaise diagnose?**
   A: (1) Run perplexity benchmark on calibration set (compare FP16 vs quantized), (2) Check if drop is on specific task types (long context? math? code?), (3) Try larger group_size (128→64), (4) Try AWQ if using GPTQ, (5) Consider INT8 instead of INT4 if quality critical, (6) For specific tasks: fine-tune with QLoRA on quantized model.

4. **Q: FP8 vs INT8 — kab kya?**
   A: FP8: H100/H200 only (hardware support), near-zero quality loss, no calibration needed, best for training + inference. INT8 (SmoothQuant): Works on A100/A10G, needs calibration, slight quality loss, wider hardware support. If you have H100 → FP8 always. If A100 → INT8 with SmoothQuant or just use INT4 AWQ.
