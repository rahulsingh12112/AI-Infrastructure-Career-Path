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

### Elastic Fabric Adapter for AI/ML and HPC workloads on Amazon EC2

Layer 1: EFA Hardware (NIC)
Physical device — actual network card jo RDMA karta hai. Ye wire pe data bhejta/receive karta hai.

Layer 2: EFA Driver (Kernel module)
OS ke andar loaded module jo hardware ko control karta hai — jaise NIC ko initialize karna, memory register karna, queues setup karna.

Layer 3: Libfabric (Common abstraction layer)
Ye KEY layer hai. Problem ye hai ki duniya mein alag-alag network fabrics hain — EFA, InfiniBand, Intel OFI, etc. Har ek ka apna API hai.

Libfabric ek universal translator hai — upar wali libraries (NCCL/MPI) ko bas ek hi API sikhni padti hai, neeche chahe koi bhi hardware ho.

NCCL → Libfabric → EFA       (AWS pe)
NCCL → Libfabric → InfiniBand (on-prem pe)


Bina Libfabric ke, NCCL ko har hardware ke liye alag code likhna padta.

Layer 4: Communication Libraries (NCCL / MPI / NIXL)

| Library | Kya karta hai | Kaun use karta hai |
|---------|--------------|-------------------|
| NCCL | GPU-to-GPU collective ops (all-reduce, broadcast) | ML training (PyTorch, TensorFlow) |
| MPI | General-purpose distributed computing | Traditional HPC (simulations, weather models) |
| NIXL | Inference ke beech data transfer | Inference workloads (model serving) |

Layer 5: Application (Training Script)
Tumhara actual Python code — model.fit() ya torch.distributed.launch. Isko neeche ki kisi layer ki tension nahi, bas NCCL/MPI call karta hai.

Ye simply bol raha hai ki tumhara training code (jaise PyTorch/TensorFlow script) ko networking ki koi chinta nahi — wo bas torch.distributed ya model.fit() call karta hai, aur andar se 
NCCL automatically sab GPU communication handle kar leta hai.

Tumhe ye jaanne ki zaroorat nahi ki data EFA se ja raha hai ya InfiniBand se, RDMA ho raha ya nahi — sab neeche ki layers apne aap manage karti hain. Tum sirf apna ML code likho.

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

---

#### NCCL / MPI / NIXL — Communication Libraries (Deep Dive)

**Ye teen libraries Libfabric ke upar baithi hain aur actual communication patterns handle karti hain.**

**NCCL (NVIDIA Collective Communications Library):**
- Multiple GPUs ke beech data share karta hai — training mein gradients sync karna
- GPU-to-GPU communication (same machine + across machines dono)
- Example: 4 EC2 × 8 GPU = 32 GPUs, sab NCCL se sync karte hain
- Same EC2 ke andar → NVLink/NVSwitch se baat
- EC2-to-EC2 → EFA (via Libfabric) se baat
- Kaun use karta hai: PyTorch, TensorFlow, DeepSpeed, Megatron

**NCCL Operations:**
| Operation | Kya karta hai | Analogy |
|-----------|--------------|---------|
| AllReduce | Sabka sum/avg lo, sabko de do | Group mein sabne marks bataye, sabko average mila |
| Broadcast | Ek GPU se sabko same data bhejo | Teacher ne sabko ek hi paper diya |
| AllGather | Sabne apna data diya, sabko sab ka mila | WhatsApp group mein sabne photo daali, sabko sab ki mili |
| ReduceScatter | Sum lo aur todke baant do | Bill split karna |

**MPI (Message Passing Interface):**
- Wahi kaam jo NCCL karta hai, but CPUs ke liye — same machine ke CPUs aur alag machines ke CPUs ke beech data share karna
- Traditional HPC ka standard (weather models, physics simulations)
- CPU-focused, purani duniya

**NIXL (NVIDIA Inference Xfer Library):**
- Jab ek bada model ek machine mein fit nahi hota aur inference ke time ek machine ka output dusri machine ko bhejna hota hai — woh fast transfer NIXL karta hai
- Training nahi, sirf inference ke liye
- NCCL training pattern (AllReduce — sab sync karo) ke liye optimized hai, NIXL inference pattern (pipeline — ek ka output next ko, low latency) ke liye

**Important:** Ek time pe ek hi use hoga — training kar rahe ho toh NCCL, inference kar rahe ho toh NIXL. Dono saath mein nahi chalte.

**GPU Instance Mein Kitne GPUs?**
- Depend karta hai instance type pe — 1 GPU (G4dn.xlarge) se lekar 8 GPUs (P4d, P5) tak ek single EC2 mein
- 8 GPU samjho jaise ek server mein 8 CPU sockets — bas GPU parallel compute karte hain

**Complete Flow:**
```
ML Training (PyTorch):
  model.backward()      ← Gradient calculate hua
  NCCL.all_reduce()     ← Sab GPUs ne gradient share kiya
  Libfabric             ← Network abstraction
  EFA/InfiniBand        ← Actual wire pe data gaya

HPC Job (Weather Simulation):
  calculate_zone()      ← Apna zone compute kiya
  MPI_Send/Recv()       ← Boundary data share kiya
  Libfabric             ← Network abstraction
  EFA/InfiniBand        ← Actual wire pe data gaya

Inference (LLaMA serving):
  forward_pass()        ← Apne layers process kiye
  NIXL.transfer()       ← Next machine ko output bheja
  Libfabric             ← Network abstraction
  EFA/InfiniBand        ← Actual wire pe data gaya
```

**Interview Line:**
> "NCCL handles GPU collective communication for training workloads, MPI is the traditional standard for distributed HPC computing, and NIXL is NVIDIA's newer library optimized for inference data transfers. All three sit on top of Libfabric which abstracts the underlying network hardware."

---

#### EFA Device Attach — 2 Tarike

**ENA vs EFA Kya Hai:**
```
ENA = Normal sadak (regular road) — IP traffic, internet, VPC, SSH sab yahan se jaata hai
EFA = Highway/Expressway — sirf heavy GPU traffic ke liye, ultra fast, no OS kernel involvement
```

**Option 1: EFA with ENA (dono saath mein) — Most Common:**
```
┌─────────────────────────┐
│      EC2 Instance        │
│                          │
│  SSH traffic ──→ ENA ──→ Internet/VPC (normal IP networking)
│  GPU training ──→ EFA ──→ Other GPU nodes (RDMA, fast)
│                          │
│  [Same physical NIC, 2 logical devices]
└─────────────────────────┘
```
Kab: Tumhe SSH bhi karna hai, internet bhi chahiye, AUR GPU communication bhi chahiye.

**Option 2: EFA-only (sirf highway, no sadak):**
```
┌─────────────────────────┐
│      EC2 Instance        │
│                          │
│  ENA NIC ──→ Internet/VPC (SSH, management)
│  EFA-only NIC ──→ SIRF GPU traffic (no IP, no internet)
│                          │
│  [Alag dedicated NIC sirf ML traffic ke liye]
└─────────────────────────┘
```
Kab: Dedicated NIC chahiye sirf GPU-to-GPU ke liye — IP address nahi chahiye, bas raw speed.

**EFA Ka OS-Bypass:**
```
Normal (ENA):  App → System Call → Kernel → TCP/IP → NIC Driver → NIC → Wire
                        (slow: context switch, packet copy, interrupt)

EFA:           App → Libfabric (user-space) → EFA NIC → Wire
                        (fast: kernel skip, no copy, no interrupt)
```
Jaise FASTag se toll booth pe bina ruke nikalte ho — EFA mein data bina OS ke ruke nikalti hai.

ENA with EFA = 1 ENI (ek hi NIC pe dono kaam — IP traffic bhi, RDMA bhi), EFA-only = 2 ENI (ek ENA wali normal ke liye, ek alag EFA-only dedicated sirf GPU traffic ke liye).

EFA-only NIC pe IP address nahi hota — wo sirf raw RDMA/SRD traffic ke liye hai, koi IP networking nahi (no SSH, no internet, no VPC routing). ENA wali ENI pe IP rehta hai — management, 
SSH, internet sab usi se hota hai.

SRD ek AWS-proprietary transport protocol hai jo EFA NIC ke through EC2 instances ke GPUs ke beech ultra-fast RDMA-like communication karta hai — ye InfiniBand/RoCE ka cloud replacement 
hai, unke beech communicate nahi karwata.

correct — NCCL/MPI/NIX (ML frameworks) → Libfabric (common API) → SRD (transport protocol) → EFA NIC (hardware) — upar se neeche yahi flow hai.

**SRD Protocol (AWS Ka Custom Protocol):**
- Normal RDMA protocols (InfiniBand, RoCE) AWS ke shared network pe directly nahi chal sakte
- AWS ne apna banaya: SRD (Scalable Reliable Datagram)
- Reliable (data loss nahi), congestion control built-in, multi-path (kai raaste ek saath)
- On-prem mein InfiniBand protocol hai, AWS mein SRD protocol — kaam same, bas cloud ke liye designed

**Comparison Table:**
| | ENA | EFA (with ENA) | EFA-only |
|--|-----|---------------|----------|
| Kya hai | Normal NIC | Normal + Fast NIC | Sirf Fast NIC |
| IP address | ✅ | ✅ | ❌ |
| SSH/Internet | ✅ | ✅ | ❌ |
| RDMA/OS-bypass | ❌ | ✅ | ✅ |
| Analogy | Regular road | Road + Highway combo | Sirf highway, no exit |
| Use case | Web apps | GPU training (most common) | Dedicated ML traffic isolation |

**Typical Setup:**
```
P4d/P5 instance:
  Primary NIC: ENA → SSH, management, S3 download, internet
  Secondary NIC: EFA (with ENA) → GPU-to-GPU training traffic
```

**Interview Line:**
> "EFA provides OS-bypass networking using AWS's SRD protocol, giving RDMA-like performance over the shared AWS network. You can attach it alongside ENA for combined IP + high-speed traffic, or use EFA-only for dedicated ML communication without IP overhead."

#### EFA — Supported Libraries, Instances, OS, Limitations & Pricing

**Supported Libraries:**
| Library | Version | Use Case |
|---------|---------|----------|
| Open MPI | 4.1+ | HPC applications |
| Intel MPI | 2019 Update 5+ | HPC applications |
| NCCL | 2.4.2+ | ML training (PyTorch, TensorFlow) |
| NIXL | 1.0.0+ | Disaggregated inference |
| AWS Neuron SDK | 2.3+ | Trainium/Inferentia chips |

**Supported Instance Types (Key ones — EFA v4, Nitro v6, RDMA read + write):**
| Category | Example Instances |
|----------|-----------------|
| General Purpose | m8a.48xlarge, m8i.96xlarge, m9g.48xlarge |
| Compute Optimized | c8a.48xlarge, c8i.96xlarge, c9g.48xlarge |
| Memory Optimized | r8a.48xlarge, r8i.96xlarge, x8i.96xlarge |
| Storage Optimized | i8ge.48xlarge |
| Accelerated (GPU) | g7.48xlarge, g7e.48xlarge, p6-b200.48xlarge, p6-b300.48xlarge |
| HPC | hpc8a.96xlarge |

> Note: Mostly large/metal instances hi EFA support karte hain — chhote sizes mein nahi milta.

**Region-wise check karne ka command:**
```bash
aws ec2 describe-instance-types \
    --region us-east-1 \
    --filters Name=network-info.efa-supported,Values=true \
    --query "InstanceTypes[*].[InstanceType]" \
    --output text | sort
```

**Supported Operating Systems:**
| OS | Intel/AMD (x86_64) | Graviton (arm64) |
|----|-------------------|-----------------|
| Amazon Linux 2023 | ✅ | ✅ |
| RHEL 8, 9, 10 | ✅ | ✅ |
| Debian 11, 12, 13 | ✅ | ✅ |
| Rocky Linux 8, 9 | ✅ | ✅ |
| Ubuntu 22.04, 24.04, 26.04 | ✅ | ✅ |
| SUSE Linux 15 SP2+ | ✅ | ✅ |

**EFA Limitations (Kya NAHI kar sakta):**
| Limitation | Explanation |
|-----------|-------------|
| RDMA write sab instances pe nahi | Kuch purane instances sirf RDMA read support karte hain |
| P4d/P4de/DL1 ↔ other instances ke beech EFA traffic nahi | Sirf same type ke beech kaam karega |
| Multiple NIC support | Multiple network cards wale instances mein — ek EFA per card. Baaki instances mein sirf 1 EFA |
| Same AZ mandatory | EFA traffic sirf same AZ mein — dono nodes same AZ mein hone chahiye |
| Same VPC mandatory | EFA traffic ek VPC se doosre VPC mein nahi ja sakta |
| Non-routable | EFA traffic route nahi hota (jaise normal IP packets). Ye direct point-to-point hai |
| AWS Outposts pe nahi | On-prem Outposts mein supported nahi |
| Windows pe limited | Windows mein sirf AWS CDI SDK applications ke liye. Baaki ke liye sirf ENA jaisa behave karega |
| Dedicated Hosts/Instances | c7g.16xl, m7g.16xl, r7g.16xl pe dedicated mein supported nahi |

> Important: EFA traffic = same AZ, same VPC, non-routable. Ye ek "private fast lane" hai — internet ya cross-region ke liye ENA/normal networking use karo.

**EFA Pricing:**
FREE hai — koi extra cost nahi. Supported instance pe enable karo, koi additional charge nahi lagta. Sirf instance ka cost dete ho.

**Summary — Ek Nazar Mein:**
```
EFA = AWS ka high-speed NIC jo OS bypass karke direct GPU-to-GPU communication deta hai
    → SRD protocol use karta hai (congestion control built-in)
    → Libfabric ke through NCCL/MPI connect hote hain
    → Same AZ, same VPC mein kaam karta hai
    → Free hai, sirf supported instance chahiye
    → ML training aur HPC ke liye essential
```

### Reference link - https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html
---

#### Placement Groups
Cluster placement group mein GPU nodes physically paas mein hote hain — lowest possible network latency milti hai nodes ke beech.

**Kya hai Placement Group?**
Placement group ek tarika hai jisme tum apne interdependent EC2 instances ko ek group mein launch karte ho taaki unki physical placement ko influence kar sako. Basically AWS ko bol rahe ho: "Mere instances ko physically KAISE arrange karo hardware pe — ye main decide karunga based on meri application ki need."

---

**Real Data Center Mein Kya Hota Hai?**

AWS ka ek **Region** hota hai (e.g., ap-south-1 = Mumbai). Uske andar multiple **Availability Zones** hoti hain (ap-south-1a, ap-south-1b, etc.). Har AZ = ek ya zyada physical data centers.

```
Data Center
├── Row 1
│   ├── Rack 1 (apna power supply, network switch)
│   │   ├── Server 1
│   │   ├── Server 2
│   │   └── Server 3
│   ├── Rack 2
│   │   ├── Server 4
│   │   ├── Server 5
│   │   └── Server 6
│   └── ...
├── Row 2
│   ├── Rack 3
│   ├── Rack 4
│   └── ...
└── ...
```

**Rack** = ek physical cabinet jisme multiple servers hote hain. Har rack ka apna:
- Power supply (UPS)
- Top-of-Rack (ToR) network switch
- Cooling

**Agar ek rack fail hua** (power gone, switch dead) → us rack ke **saare servers** down.

---

**4 Placement Strategies:**

**1. CLUSTER — "Sab ek saath chipka do"**

Jaise ek table pe saare laptops ek doosre ke bilkul paas rakh do.

```
Availability Zone: ap-south-1a
└── Data Center X
    └── Row 1
        └── Rack 1 (same ToR switch)
            ├── Instance A  ← Cluster group
            ├── Instance B  ← Cluster group
            ├── Instance C  ← Cluster group
            └── Instance D  ← Cluster group
```

- Saare instances **same rack ya adjacent racks** mein hote hain
- Same Top-of-Rack switch se connected → **~10 Gbps+ bandwidth, <1ms latency**
- Ek hi spine switch ke neeche → minimal network hops
- **Fayda:** Sab ek doosre se bohot fast baat kar sakte hain (low latency)
- **Use case:** HPC, tightly-coupled GPU training, MPI jobs
- **Risk:** Rack fail = **saare instances gone**. Availability sacrifice for performance.

```bash
# MPI-based HPC job — nodes ko aapas mein nanosecond-level mein baat karni hai
aws ec2 create-placement-group \
    --group-name hpc-cluster \
    --strategy cluster

# Launch with Enhanced Networking (ENA) + EFA
aws ec2 run-instances \
    --instance-type c5n.18xlarge \
    --placement "GroupName=hpc-cluster" \
    --network-interfaces "InterfaceType=efa" \
    --count 8
```

---

**2. PARTITION — "Group banao, lekin groups ko alag hardware do"**

Jaise ek class mein 4 groups bana do aur har group ko alag room mein bitha do. Ek room ka AC kharab ho toh baaki groups safe.

```
Availability Zone: ap-south-1a
└── Data Center X
    ├── Rack 1 (Partition 1)
    │   ├── Instance A
    │   ├── Instance B
    │   └── Instance C
    ├── Rack 3 (Partition 2)    ← different hardware
    │   ├── Instance D
    │   ├── Instance E
    │   └── Instance F
    └── Rack 7 (Partition 3)    ← different hardware
        ├── Instance G
        ├── Instance H
        └── Instance I
```

- Har partition ko **alag set of racks** milta hai
- Partition 1 aur Partition 2 kabhi same rack share nahi karenge
- **Max 7 partitions per AZ**
- Har partition mein multiple instances ho sakte hain (koi limit nahi)
- **Use case:** Hadoop, Kafka, Cassandra — jahan data replicated hota hai aur failure isolate chahiye

```bash
# Kafka cluster — brokers ko partition-aware deploy karo
aws ec2 create-placement-group \
    --group-name kafka-partitioned \
    --strategy partition \
    --partition-count 3

# Launch Kafka Broker 1 in Partition 1
aws ec2 run-instances \
    --instance-type r5.2xlarge \
    --placement "GroupName=kafka-partitioned,PartitionNumber=1" \
    --count 3

# Launch Kafka Broker 2 in Partition 2
aws ec2 run-instances \
    --instance-type r5.2xlarge \
    --placement "GroupName=kafka-partitioned,PartitionNumber=2" \
    --count 3
```

---

**3. SPREAD — "Har ek ko alag rack/hardware do"**

Jaise har student ko alag bench pe bitha do — ek bench toote toh baaki sab safe.

```
Availability Zone: ap-south-1a
└── Data Center X
    ├── Rack 1
    │   └── Instance A    ← spread group
    ├── Rack 4
    │   └── Instance B    ← spread group
    ├── Rack 9
    │   └── Instance C    ← spread group
    └── ...

Availability Zone: ap-south-1b
└── Data Center Y
    ├── Rack 2
    │   └── Instance D    ← spread group
    └── Rack 6
        └── Instance E    ← spread group
```

- **Har instance guaranteed alag rack** pe hoga
- Limit: **max 7 instances per AZ** (kyunki guaranteed isolation deni hoti hai)
- Multi-AZ spread ho sakta hai
- **Use case:** ZooKeeper ensemble, etcd cluster, critical control plane nodes

```bash
# ZooKeeper ensemble — 5 nodes, har ek alag hardware pe
aws ec2 create-placement-group \
    --group-name zk-spread \
    --strategy spread

aws ec2 run-instances \
    --instance-type m5.xlarge \
    --placement "GroupName=zk-spread" \
    --count 5
```

**Risk:** Performance sacrifice — instances door hain toh latency thodi zyada. But **maximum fault isolation**.

---

**4. PRECISION TIME — "Exact time chahiye"**

Jaise ek special ghari wali seat pe bitha do jahan atomic clock lagti hai.

```
Availability Zone: ap-south-1a
└── Data Center X
    └── Special Rack (with PTP-capable hardware)
        ├── Instance A    ← precision time group
        └── Instance B    ← precision time group
```

- AWS ke specific servers mein **PTP (Precision Time Protocol) hardware** laga hota hai
- Ye servers directly **atomic clock / GPS-synchronized time source** se connected hain
- Normal instances ko NTP milta hai (~millisecond accuracy)
- Precision time instances ko **microsecond-level accuracy** milti hai
- **Use case:** Financial trading systems, time-sensitive distributed databases

```bash
aws ec2 create-placement-group \
    --group-name trading-time \
    --strategy precision-time

aws ec2 run-instances \
    --instance-type c5.xlarge \
    --placement "GroupName=trading-time" \
    --count 2
```

---

**Comparison Table — Data Center Perspective:**

| Strategy | Hardware Placement | Failure Blast Radius | Network Latency | Limit |
|----------|-------------------|---------------------|-----------------|-------|
| Cluster | Same rack / adjacent racks | HIGH (ek rack = sab gone) | Lowest (~μs) | No instance limit |
| Partition | Different rack sets per partition | MEDIUM (ek partition affected) | Medium | 7 partitions/AZ |
| Spread | Each instance on different rack | LOWEST (sirf 1 instance affected) | Higher | 7 instances/AZ |
| Precision Time | PTP-capable hardware racks | Depends | Normal | Hardware availability |

---

**Network Topology Context:**

```
Internet
  │
  ▼
AWS Backbone
  │
  ▼
Region (ap-south-1)
  ├── AZ-a ──── Data Center(s)
  │               ├── Spine Switch
  │               │   ├── ToR Switch (Rack 1) ← Cluster instances yahan
  │               │   ├── ToR Switch (Rack 2)
  │               │   └── ToR Switch (Rack 3)
  │               └── ...
  ├── AZ-b ──── Data Center(s)
  └── AZ-c ──── Data Center(s)
```

- **Cluster:** Same ToR switch ke neeche = minimum hops = fastest
- **Spread:** Different ToR switches = zyada hops = thoda slow but isolated
- **Partition:** Beech ka — groups isolated, within group fast

---

**Pricing:**
**FREE** — placement group banana ka koi charge nahi.

**Rules & Limitations:**
1. Ek instance sirf ek placement group mein ho sakta hai — multiple placement groups mein nahi
2. Placement groups merge nahi ho sakte
3. On-Demand Capacity Reservations aur zonal Reserved Instances — agar instance attributes match karte hain toh reserved capacity automatically use hoti hai, chahe instance placement group mein ho ya nahi
4. Dedicated Hosts ko placement groups mein launch nahi kar sakte
5. Spot Instance jo stop ya hibernate on interruption ke liye configured hai, usse placement group mein launch nahi kar sakte

**Interview Line:**
> "Placement groups let you control the physical placement of EC2 instances on underlying hardware. Cluster strategy gives lowest latency by co-locating on same rack (ideal for EFA + NCCL GPU training), Partition isolates failure domains for distributed systems like Kafka/HDFS, and Spread maximizes fault isolation by placing each instance on separate hardware."

---

#### Placement Strategies — Deep Dive (Har Strategy Ki Poori Detail)

---

**1. CLUSTER Placement Group — Deep Dive**

**Ek line:** Saare instances ek hi AZ mein paas-paas bitha do — fastest network speed ke liye.

**Key points:**

- Instances **single AZ** mein hote hain, but **ek hi rack tak limited nahi** — nearby racks mein bhi ho sakte hain
- **Peered VPCs span kar sakta hai** same Region mein — matlab do alag VPCs ke instances bhi ek cluster group mein aa sakte hain
- **Higher per-flow throughput:** Cluster mein = 10 Gbps single-flow, bina cluster = 5 Gbps single-flow
- **High-bisection bandwidth segment** mein place hote hain — network ka wo part jahan maximum bandwidth available hai

**Precision Time + Cluster combo:**
- Cluster group banate waqt `--parent-group-id` se ek precision time group attach kar sakte ho
- Isse cluster ke instances ko **fast network + accurate time dono** mil jaata hai

**Best practices:**
- **Ek hi launch request** mein saare instances launch karo — baad mein add karoge toh capacity error aa sakta hai
- **Same instance type** use karo sab mein — mixed types = zyada chance of capacity error
- Agar capacity error aaye → **sab instances stop karo, phir start karo** — AWS migrate kar dega naye hardware pe jahan sab fit ho jayein

**Stop/Start behavior:**
- Instance stop karke start kiya → **wahi placement group mein rehta hai**
- But agar capacity nahi hai toh start FAIL hoga

**Rules & Limitations:**

| Rule | Detail |
|------|--------|
| AZ | Sirf 1 AZ — multiple AZ span nahi kar sakta |
| Supported instances | Current gen (except T2, Mac1, M7i-flex) + some old gen (A1, C3, C4, I2, M4, R3, R4) |
| Throughput limit | Do instances ke beech speed = **slower instance ki speed** se limited |
| Enhanced networking | Cluster mein = 10 Gbps/flow, bina cluster = 5 Gbps/flow |
| S3 traffic | Same region S3 → full aggregate bandwidth use kar sakta hai (no 10 Gbps limit) |
| Multiple instance types | Allowed hai, but **recommended nahi** — capacity error ka risk |
| Capacity reservation | **On-Demand Capacity Reservation** se reserve karo cluster mein. Zonal Reserved Instances se placement group mein explicitly reserve NAHI kar sakte |
| Internet/Direct Connect | **5 Gbps limit** for traffic going to internet or on-prem via Direct Connect |

**Simple analogy:** Jaise ek office mein sab developers ko ek hi floor pe bitha do — communication fastest, but agar fire lage toh sab affected.

---

**2. PARTITION Placement Group — Deep Dive**

**Ek line:** Instances ko groups (partitions) mein divide karo, har group ko alag hardware (racks) do — ek group fail hua toh baaki safe.

**Key points:**

- EC2 har partition ko **apna set of racks** deta hai — apna power, apna network
- **Koi do partition same rack share nahi karte** — hardware failure isolation guaranteed
- AWS try karta hai instances ko **evenly distribute** kare across partitions — but **guarantee nahi** hai even distribution ki
- Tum khud **specific partition mein** instance launch kar sakte ho (PartitionNumber specify karke)
- **Partition visibility** milti hai — tum dekh sakte ho ki kaunsa instance kaunse partition mein hai

**Ye visibility kyun important hai:**
- HDFS/Cassandra/HBase jaise apps ko pata hota hai ki kaunsa replica kahan hai
- Toh wo **intelligent replication decisions** le sakte hain — ek replica Partition 1 mein, doosra Partition 2 mein → agar Partition 1 fail hua toh data safe

**Multi-AZ support:**
- Partitions **multiple AZs mein** ho sakte hain same Region mein
- Max **7 partitions per AZ**

**Capacity error handling:**
- Agar unique hardware nahi mila → request fail
- Baad mein try karo — AWS over time more hardware available karata hai

**Rules & Limitations:**

| Rule | Detail |
|------|--------|
| Max partitions | 7 per AZ |
| Max instances | Account limit tak — koi partition-level instance limit nahi |
| Even distribution | AWS tries, but **guaranteed nahi** |
| Dedicated Instances | Max **2 partitions** allowed (normal mein 7) |
| Capacity Reservations | **Reserve NAHI** hota partition placement group mein |

**Simple analogy:** Jaise school mein 3 alag buildings hain — ek building mein light gayi toh baaki 2 buildings ke bachchon ki class normal chalti hai.

---

**3. SPREAD Placement Group — Deep Dive**

**Ek line:** Har ek instance ko guaranteed alag hardware (rack) pe rakh do — maximum isolation.

**Key points:**

- Har instance **distinct hardware** pe — koi sharing nahi
- **Small number of critical instances** ke liye best — jaise master nodes, ZooKeeper, etcd
- **Mix instance types** allowed — time ke saath alag alag launch kar sakte ho
- Capacity nahi mili → fail, baad mein try karo

**2 Levels of Spread:**

| Level | Kahan available | Kya karta hai |
|-------|----------------|---------------|
| **Rack level** | AWS Regions + Outposts | Har instance alag rack pe |
| **Host level** | Sirf AWS Outposts | Har instance alag physical host pe |

**Rack level spread:**
- **Max 7 running instances per AZ per group**
- Multiple AZ span kar sakta hai — e.g., 3 AZs × 7 = **21 instances total** ek group mein
- Outposts mein — jitne racks hain utne instances rakh sakte ho

**8th instance chahiye same AZ mein?**
- **Naya spread placement group** banao
- But do alag groups ke beech spread ki **koi guarantee nahi** — sirf within-group guarantee hai

**Host level spread:**
- Sirf Outposts pe
- Jitne hosts hain utne instances

**Rules & Limitations:**

| Rule | Detail |
|------|--------|
| Max instances/AZ | 7 per AZ (rack level, in Region) |
| Multi-AZ | Haan — multiple AZ span kar sakta hai |
| Dedicated Instances | **Supported NAHI** |
| Host level | Sirf Outposts pe |
| Capacity Reservations | **Reserve NAHI** hota spread placement group mein |
| Multiple groups | Allowed, but inter-group spread guaranteed nahi |

**Simple analogy:** Jaise 7 important files ko 7 alag lockers mein rakh do — ek locker toota toh baaki 6 safe.

---

**4. PRECISION TIME Placement Group — Deep Dive**

**Ek line:** Instances ko aisi hardware pe rakh do jahan ultra-accurate time milta hai — microsecond level.

**Key points:**

- Enhanced Amazon Time Sync Service access milta hai
- Normal instances ko standard NTP milta hai — ye **better accuracy** deta hai

**3 Benefits:**

| Benefit | Detail | OS Support |
|---------|--------|-----------|
| Enhanced local NTP | Better accuracy NTP source, no config needed | All supported OS |
| PTP Hardware Clock (PHC) | Microsecond-accurate sync via hardware device | Linux only |
| Hardware packet timestamping | Nanosecond-resolution network measurements | Linux only |

**Use cases:**
- Distributed databases jahan **transaction ordering** chahiye (kaunsa transaction pehle aaya)
- Financial services — **precise timestamping** (trade execution time)
- Distributed systems jo **local clock readings** se events order karte hain

**Cluster + Precision Time combo:**
- Pehle precision time group banao
- Phir cluster group banao with `--parent-group-id` = precision time group
- Result: **Fast network (cluster) + Accurate time (precision) dono**
- Important: Precision time group jo parent hai cluster ka — **delete nahi kar sakte** jab tak cluster group exists

**Rules & Limitations:**

| Rule | Detail |
|------|--------|
| Supported instances | Specific families only — docs check karo |
| Capacity error | Hardware nahi mili → fail, try later or different AZ |
| Parent group delete | Cluster ka parent precision group = delete blocked |
| Pricing | **FREE — koi extra charge nahi** |

**Simple analogy:** Jaise exam hall mein kuch special seats hain jahan wall clock bilkul sahi time dikhati hai — un seats pe baithe students ko exact time pata hota hai.

---

**Master Comparison Table:**

| | Cluster | Partition | Spread | Precision Time |
|--|---------|-----------|--------|---------------|
| **Goal** | Speed | Failure isolation (groups) | Failure isolation (individual) | Accurate time |
| **AZ span** | ❌ Single AZ only | ✅ Multi-AZ | ✅ Multi-AZ | Per AZ |
| **VPC span** | ✅ Peered VPCs | ❌ | ❌ | ❌ |
| **Max units** | No limit | 7 partitions/AZ | 7 instances/AZ | Hardware dependent |
| **Instance limit** | Account limit | Account limit | 7/AZ | Hardware |
| **Capacity Reservation** | ✅ On-Demand CR works | ❌ Nahi kaam karta | ❌ Nahi kaam karta | N/A |
| **Dedicated Instances** | Normal | Max 2 partitions | ❌ Not supported | N/A |
| **Pricing** | Free | Free | Free | Free |

---

**Capacity Error Ka Universal Rule (Sab pe apply hota hai):**

> Agar AWS ke paas tumhari request ke liye sufficient unique/appropriate hardware nahi hai → **launch/start FAIL hoga**. Solution: baad mein try karo, ya different AZ try karo. AWS over time more hardware add karta rehta hai.

---

**One Final Summary:**

```
Tumhe SPEED chahiye          → Cluster (same AZ, paas-paas, 10 Gbps/flow)
Tumhe GROUP ISOLATION chahiye → Partition (7 groups/AZ, har group alag racks)
Tumhe PER-INSTANCE ISOLATION  → Spread (har ek alag rack, max 7/AZ)
Tumhe ACCURATE TIME chahiye   → Precision Time (microsecond clock)
Tumhe SPEED + TIME dono       → Precision Time as parent → Cluster as child
```

Sab FREE hain. Koi extra charge nahi. Bas sahi instance type choose karo aur capacity available honi chahiye.

### Reference link - https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/placement-groups.html
### Reference link - https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/placement-strategies.html

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

---

## Resources
- [AWS GPU instances](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [EFA documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [NCCL documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [SageMaker distributed training](https://docs.aws.amazon.com/sagemaker/latest/dg/distributed-training.html)
- [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)
