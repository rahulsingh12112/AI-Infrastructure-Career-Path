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

#### CPU vs GPU — Basic Difference
- **CPU:** Computer ka "brain" — general-purpose calculations, logic, decisions, sequentially ek ek task handle karta hai.
- **GPU:** Hazaaron chhote cores jo parallel mein kaam karte hain — graphics rendering, ML training, aur massive math operations ke liye best hai. Originally graphics (pixels) render karne ke liye banaya gaya tha, isliye naam "Graphics Processing Unit" raha.

---

#### Normal TCP Networking vs EFA with RDMA

**Normal TCP (slow path):**
```
GPU Node A → OS → TCP Stack → Network → TCP Stack → OS → GPU Node B
               ↑ bahut saare layers, har layer pe latency
```
- CPU har step pe involved hai
- Data multiple baar copy hota hai (user space → kernel space → NIC buffer)
- System calls, context switches, interrupts — sab latency add karte hain
- Latency: ~50-100+ μs

**EFA with RDMA (fast path):**
```
GPU Node A → NIC(EFA) → Network → NIC(EFA) → GPU Node B
               ↑ beech mein koi OS/CPU involvement nahi (almost)
```
- NIC khud CPU ka kaam (data copy, protocol processing) apne upar le leta hai
- Directly ek node ki GPU memory se doosre node ki GPU memory tak data pahunchata hai
- Latency: ~1-5 μs (10-50x faster)

---

#### RDMA — How It Works (Memory to Memory)

**Setup phase (ek baar):**
1. Dono nodes ek connection establish karte hain
2. Dono apni memory ka ek region **register** karte hain NIC ke saath — "ye area tujhe directly access karne ki permission hai"
3. NIC ko remote machine ki memory ka address bataya jaata hai

**Data transfer phase (baar baar, ultra fast):**
1. Node A ka NIC khud registered memory se data uthata hai (CPU ko bina bataye)
2. Wire pe bhejta hai
3. Node B ka NIC seedha registered memory mein likh deta hai (CPU ko bina bataye)

**Ye possible kaise hai:**
| Technology | Role |
|-----------|------|
| Smart NIC (HCA) | NIC ke andar dedicated silicon hai jo protocol processing hardware-level pe karta hai |
| Memory Registration | OS ek baar memory pages ko "pin" karta hai aur NIC ko physical address deta hai |
| DMA (Direct Memory Access) | NIC hardware-level pe RAM read/write kar sakta hai bina CPU interrupt ke |
| Queue-based model | App "send queue" mein request daalta hai, NIC khud process karta hai |

**GPUDirect RDMA (most optimized):**
```
GPU A VRAM → NIC(EFA) → Network → NIC(EFA) → GPU B VRAM
```
RAM bhi skip — seedha GPU memory to GPU memory transfer.

**Comparison:**
| | TCP | RDMA |
|--|-----|------|
| Latency | ~50-100+ μs | ~1-5 μs |
| CPU usage | High | Near zero |
| Data copies | 3-4 | 0 (zero copy) |
| OS involvement | Every packet | Setup only |

---

#### EFA (Elastic Fabric Adapter)
AWS ka custom network interface jo RDMA-like capabilities deta hai cloud mein. Multi-node GPU training ke liye essential.

#### NCCL (NVIDIA Collective Communications Library)
NVIDIA ki library jo multi-GPU communication handle karti hai — gradient sync, all-reduce operations etc. EFA ke saath milke kaam karti hai.

---

#### Libfabric — The Universal Network Driver

**One Line:** Libfabric ek universal driver (library) hai jo kisi bhi network hardware (NIC) se baat kar sakta hai, taaki upar wali applications ko hardware change hone pe apna code na badalna pade.

**Problem Without Libfabric (M × N problem):**
```
NCCL ko likhna padta:
├── EFA ke liye alag code
├── InfiniBand ke liye alag code
├── Intel OFI ke liye alag code
├── RoCE ke liye alag code

MPI ko bhi likhna padta:
├── EFA ke liye alag code
├── InfiniBand ke liye alag code
├── ... same pain repeat

Har application × Har hardware = explosion of code
5 applications × 4 hardware = 20 integrations ❌
```

**Solution With Libfabric (M + N problem):**
```
┌──────┐  ┌──────┐  ┌──────┐
│ NCCL │  │ MPI  │  │ NIXL │    ← Applications
└──┬───┘  └──┬───┘  └──┬───┘
   └─────────┼─────────┘
             │
      ┌──────▼──────┐
      │  Libfabric  │              ← Universal abstraction layer
      └──────┬──────┘
             │
   ┌─────────┼─────────┐
┌──▼──┐ ┌───▼──┐ ┌───▼──┐ ┌────────┐
│ EFA │ │ IBV  │ │ RoCE │ │ Intel  │  ← Hardware providers
└─────┘ └──────┘ └──────┘ └────────┘

Total integrations = 3 + 4 = 7 ✅
```

**What are NCCL, MPI, NIXL?**
- **NCCL** — NVIDIA ki library jo multiple GPUs ke beech data share karti hai (training mein gradients sync karna). GPU-to-GPU communication.
- **MPI** — Traditional HPC ki library jo multiple machines ke beech messages bhejti hai. CPU-focused, purani duniya ka standard.
- **NIXL** — NVIDIA ki nayi library jo inference mein model parts alag machines pe hain toh unke beech data transfer karti hai.

**Libfabric Ke Andar — Providers (Plugins):**
```
┌────────────────────────────────────┐
│          Libfabric API             │
│  (fi_send, fi_recv, fi_read,      │
│   fi_write, fi_atomic, etc.)      │
├──────────┬──────────┬─────────────┤
│ Provider │ Provider │ Provider    │
│   EFA    │  Verbs   │  Sockets    │
│          │(InfiniBand)│ (fallback)│
└──────────┴──────────┴─────────────┘
```
- `efa` provider → AWS EFA hardware
- `verbs` provider → InfiniBand (libibverbs)
- `sockets` provider → Regular TCP (fallback, slow but works everywhere)

**Analogy:** Libfabric = USB standard jaisa. Jaise USB ke baad har printer ek hi cable se chalta hai, waise Libfabric ke baad har application ek hi API se kisi bhi NIC pe chalti hai.

**Networking Analogy:** SDN Controller jaisa — jaise SDN mein application ko switch vendor (Cisco/Juniper/Arista) ki chinta nahi hoti, waise NCCL ko network hardware ki chinta nahi hoti.

**Kyu AWS Ne Libfabric Use Kiya:**
1. NCCL already Libfabric support karta tha (InfiniBand ke liye)
2. AWS ne bas ek naya "efa provider" plugin likha
3. NCCL bina modification ke EFA pe chal gaya
4. Sirf 1 plugin likhne se saare applications (NCCL, MPI, NIXL) automatically EFA pe support ho gayi

**Portability Benefit:**
```
Same NCCL code kahi bhi chalega:
- AWS pe → Libfabric automatically efa provider pick karega
- On-prem pe → Libfabric automatically verbs (InfiniBand) provider pick karega
- Laptop pe testing → Libfabric automatically sockets (TCP) provider use karega
Application code SAME rehta hai!
```

**Key Takeaway:** NCCL/MPI/NIXL → Libfabric se baat karti hain → Libfabric ke andar providers (plugins) hain → jo specific hardware ke driver (EFA, InfiniBand, RoCE) se baat karte hain. Yeh sirf GPU nahi, CPU (MPI/HPC) mein bhi use hota hai — Libfabric ko farak nahi padta upar GPU hai ya CPU.

**Interview Line:**
> "Libfabric is an abstraction layer that decouples communication libraries like NCCL from the underlying network hardware. AWS wrote an EFA provider for Libfabric, which means any application already using Libfabric — NCCL, MPI, NIXL — automatically works over EFA without code changes. It converts an M×N integration problem into M+N."

#### Placement Groups
Cluster placement group mein GPU nodes physically paas mein hote hain — lowest possible network latency milti hai nodes ke beech.

#### NVLink / NVSwitch
Intra-node (same machine ke andar) GPU-to-GPU communication — PCIe se 5-10x faster bandwidth. Jaise P4d mein 8 GPUs NVSwitch se connected hain.

#### Why This Matters for Multi-Node ML Training
Distributed training mein GPUs ko continuously gradients sync karne padte hain. Agar network slow hai toh GPUs idle baithe rehte hain. EFA + RDMA + NCCL milke ye ensure karte hain ki network bottleneck na bane aur GPUs maximum utilized rahein.

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
