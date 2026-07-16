# 06 — GPU Monitoring & Troubleshooting

## Why GPU Monitoring is Critical

```
GPU cluster = expensive ($30-50/hr per A100 node)
1 hour of idle GPU = wasted $$$
1 undetected hardware error = corrupted training (days wasted)
No monitoring = flying blind
```

## Monitoring Stack Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Monitoring Stack                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  GPU Node                                                │
│  ┌──────────────────────────────┐                       │
│  │  DCGM Exporter (DaemonSet)   │                       │
│  │  ├── Collects GPU metrics     │                       │
│  │  ├── Exposes /metrics endpoint│                       │
│  │  └── Port 9400                │                       │
│  └──────────────┬───────────────┘                       │
│                 │                                         │
│                 ▼                                         │
│  ┌──────────────────────────────┐                       │
│  │  Prometheus (scrapes metrics) │                       │
│  │  ├── Stores time-series data  │                       │
│  │  ├── Alerting rules           │                       │
│  │  └── 15s scrape interval      │                       │
│  └──────────────┬───────────────┘                       │
│                 │                                         │
│                 ▼                                         │
│  ┌──────────────────────────────┐                       │
│  │  Grafana (visualization)      │                       │
│  │  ├── GPU dashboards           │                       │
│  │  ├── Real-time monitoring     │                       │
│  │  └── Alerting & notifications │                       │
│  └──────────────────────────────┘                       │
│                                                          │
│  ┌──────────────────────────────┐                       │
│  │  AlertManager                 │                       │
│  │  ├── Slack/PagerDuty alerts   │                       │
│  │  ├── GPU failure → page oncall│                       │
│  │  └── Idle GPU → cost alert    │                       │
│  └──────────────────────────────┘                       │
└─────────────────────────────────────────────────────────┘
```

## Key GPU Metrics

### Performance Metrics
```
Metric                          │ What it means              │ Healthy Range
────────────────────────────────┼────────────────────────────┼──────────────
DCGM_FI_DEV_GPU_UTIL           │ GPU compute utilization %  │ 80-100% (training)
DCGM_FI_DEV_MEM_COPY_UTIL      │ Memory bandwidth usage %   │ 20-60%
DCGM_FI_PROF_SM_ACTIVE         │ % of time SMs are active   │ 70-95%
DCGM_FI_PROF_SM_OCCUPANCY      │ Warp occupancy %           │ 50-80%
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE│ Tensor core utilization %  │ 30-80%
DCGM_FI_PROF_DRAM_ACTIVE       │ Memory active %            │ 30-70%
DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL │ NVLink bytes/sec       │ Varies
```

### Memory Metrics
```
DCGM_FI_DEV_FB_USED            │ GPU memory used (MB)       │ < 95% of total
DCGM_FI_DEV_FB_FREE            │ GPU memory free (MB)       │ > 5% of total
DCGM_FI_DEV_BAR1_USED          │ BAR1 memory used           │ Rarely an issue
```

### Health Metrics
```
DCGM_FI_DEV_GPU_TEMP           │ GPU temperature (°C)       │ < 80°C
DCGM_FI_DEV_MEMORY_TEMP        │ HBM temperature (°C)       │ < 95°C
DCGM_FI_DEV_POWER_USAGE        │ Power draw (Watts)         │ < TDP
DCGM_FI_DEV_TOTAL_ENERGY       │ Total energy consumed      │ For cost tracking
DCGM_FI_DEV_XID_ERRORS         │ Hardware error codes       │ Must be 0!
DCGM_FI_DEV_ECC_SBE_VOL_TOTAL  │ Single-bit ECC errors     │ < 5/day
DCGM_FI_DEV_ECC_DBE_VOL_TOTAL  │ Double-bit ECC errors     │ MUST be 0!
DCGM_FI_DEV_RETIRED_PAGES_SBE  │ Retired memory pages      │ Monitor trend
DCGM_FI_DEV_RETIRED_PAGES_DBE  │ Retired pages (double-bit)│ > 0 = replace GPU
```

## nvidia-smi Commands (Essential)

```bash
# Basic status
nvidia-smi

# Continuous monitoring (like top for GPUs)
nvidia-smi dmon -s pucvmet -d 1
# p=power, u=utilization, c=proc clock, v=violation, m=mem, e=ecc, t=temp

# Detailed GPU info
nvidia-smi -q -i 0              # All info for GPU 0
nvidia-smi -q -i 0 -d MEMORY   # Memory details
nvidia-smi -q -i 0 -d ECC      # ECC error details
nvidia-smi -q -i 0 -d CLOCK    # Clock speeds
nvidia-smi -q -i 0 -d POWER    # Power info

# Process list (which pod uses which GPU)
nvidia-smi pmon -d 1            # Process monitoring

# Topology
nvidia-smi topo -m              # GPU interconnect topology

# Reset GPU (if hung)
nvidia-smi -r -i 0             # Reset GPU 0 (kills all processes!)

# Set persistence mode (keeps driver loaded)
nvidia-smi -pm 1               # Reduces cold-start latency

# Clock management
nvidia-smi -ac 1215,1410 -i 0  # Set app clocks (memory, graphics)
nvidia-smi -rac                 # Reset app clocks to default
```

## DCGM (Data Center GPU Manager)

```bash
# Install DCGM
apt-get install datacenter-gpu-manager

# Run diagnostics
dcgmi diag -r 1    # Level 1: Quick (30 sec)
dcgmi diag -r 2    # Level 2: Medium (2 min)
dcgmi diag -r 3    # Level 3: Full (15+ min, stresses GPU)

# Health check
dcgmi health -c -g 0    # Check group 0 health
dcgmi health -g 0       # Show health results

# Field discovery
dcgmi dmon -e 203,204,1001,1002,1003 -d 1000
# 203=GPU Util, 204=Mem Util, 1001=SM Active, 1002=SM Occupancy, 1003=Tensor Active

# GPU groups
dcgmi group -c "training-gpus" -a 0,1,2,3    # Create group
dcgmi stats -s jobid123 -g training-gpus       # Job stats
dcgmi stats -v -j jobid123                     # Verbose job stats
```

## Common GPU Errors & How to Fix

### XID Errors (Most Important)

```
XID Error │ Meaning                    │ Severity │ Action
──────────┼────────────────────────────┼──────────┼──────────────────
XID 13    │ Graphics engine error      │ Medium   │ Reset GPU, if recurring → RMA
XID 31    │ GPU memory page fault      │ Low      │ App bug (bad pointer)
XID 38    │ Driver firmware error      │ Medium   │ Update driver
XID 43    │ GPU stopped responding     │ High     │ Reset GPU + check power/thermal
XID 45    │ Preemptive GPU reset       │ Low      │ Normal in some cases
XID 48    │ Double-bit ECC error       │ CRITICAL │ Replace GPU immediately!
XID 63    │ ECC row remapping failure  │ High     │ Schedule replacement
XID 64    │ ECC page retirement        │ Medium   │ Monitor — too many = replace
XID 79    │ GPU fell off the bus       │ CRITICAL │ PCIe issue, reseat/replace
XID 92    │ High single-bit ECC rate   │ High     │ Schedule replacement soon
XID 94    │ Contained ECC error        │ Medium   │ MIG: only partition affected
XID 95    │ Uncontained ECC error      │ CRITICAL │ Whole GPU affected, reset needed
```

### Troubleshooting Flowchart

```
GPU Problem?
├── Training slow?
│   ├── nvidia-smi: GPU Util < 50%?
│   │   ├── YES → Data loading bottleneck (CPU/storage issue)
│   │   └── NO → Check SM occupancy, memory bandwidth
│   ├── nvidia-smi: Memory near 100%?
│   │   └── Reduce batch size or use gradient accumulation
│   └── NCCL busbw low?
│       └── Check topology, network config (see file 03)
│
├── Training crashed?
│   ├── OOM (Out of Memory)?
│   │   ├── Reduce batch size
│   │   ├── Use gradient checkpointing
│   │   ├── Use mixed precision (FP16/BF16)
│   │   └── Check for memory leaks
│   ├── XID error in dmesg?
│   │   ├── XID 48/79 → Hardware failure, replace GPU
│   │   ├── XID 43 → Thermal/power issue
│   │   └── XID 13 → Try driver update, then RMA
│   └── NCCL timeout?
│       └── See file 03 for NCCL debugging
│
├── GPU not visible?
│   ├── nvidia-smi shows error?
│   │   ├── "No devices found" → Driver not loaded
│   │   ├── "GPU has fallen off the bus" → XID 79, reseat GPU
│   │   └── "Unable to determine..." → Driver mismatch
│   └── In K8s pod, GPU not allocated?
│       ├── Check device plugin running (kubectl get pods -n gpu-operator)
│       ├── Check node labels (nvidia.com/gpu.count)
│       └── Check resource limits in pod spec
│
└── GPU memory leak?
    ├── nvidia-smi shows memory used but no process?
    │   ├── Zombie CUDA process → kill -9
    │   ├── MPS server holding context → restart mps
    │   └── Driver bug → nvidia-smi -r (reset)
    └── Memory grows over time?
        ├── Python garbage collection issue
        ├── torch.no_grad() missing in eval
        └── Tensor references not released
```

## Prometheus Alerting Rules

```yaml
# GPU Alert Rules
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gpu-alerts
spec:
  groups:
  - name: gpu.rules
    rules:
    # GPU is idle (wasting money!)
    - alert: GPUIdle
      expr: DCGM_FI_DEV_GPU_UTIL < 10
      for: 30m
      labels:
        severity: warning
      annotations:
        summary: "GPU {{ $labels.gpu }} idle for 30+ minutes"
        description: "GPU utilization below 10%. Consider releasing."

    # GPU temperature too high
    - alert: GPUHighTemperature
      expr: DCGM_FI_DEV_GPU_TEMP > 83
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "GPU {{ $labels.gpu }} overheating: {{ $value }}°C"

    # ECC errors detected
    - alert: GPUECCError
      expr: DCGM_FI_DEV_ECC_DBE_VOL_TOTAL > 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "GPU {{ $labels.gpu }} has double-bit ECC errors!"
        description: "Hardware failure imminent. Replace GPU."

    # GPU memory almost full
    - alert: GPUMemoryHigh
      expr: (DCGM_FI_DEV_FB_USED / (DCGM_FI_DEV_FB_USED + DCGM_FI_DEV_FB_FREE)) > 0.95
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "GPU {{ $labels.gpu }} memory > 95% used"

    # XID errors
    - alert: GPUXIDError
      expr: DCGM_FI_DEV_XID_ERRORS > 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "GPU {{ $labels.gpu }} XID error detected: {{ $value }}"

    # NVLink errors
    - alert: NVLinkError
      expr: rate(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL[5m]) > 0
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "NVLink CRC errors on GPU {{ $labels.gpu }}"
```

## Grafana Dashboard (Key Panels)

```
Recommended Dashboard Layout:

Row 1: Overview
├── Total GPUs vs Active GPUs (gauge)
├── Cluster GPU Utilization (avg) (graph)
└── GPU Memory Usage % (heatmap)

Row 2: Per-GPU Details
├── GPU Utilization per GPU (graph, stacked)
├── GPU Memory per GPU (graph)
└── GPU Temperature (graph + threshold line)

Row 3: Training Performance
├── Tensor Core Utilization (graph)
├── SM Occupancy (graph)
└── NVLink Bandwidth (graph)

Row 4: Health
├── ECC Errors (counter)
├── XID Errors (table, last 24h)
├── Throttle Reasons (thermal/power)
└── Retired Pages count

Row 5: Cost
├── GPU Hours Used per Team (bar chart)
├── Idle GPU Hours (waste tracker)
└── Cost per Training Job (estimated)
```

## Kubernetes-Specific GPU Debugging

```bash
# Check if GPU operator is healthy
kubectl get pods -n gpu-operator
kubectl logs -n gpu-operator -l app=nvidia-device-plugin

# Check GPU resources on nodes
kubectl describe node gpu-node-1 | grep -A 5 "Allocated resources"
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity.'nvidia\.com/gpu'

# Check GPU allocation to pods
kubectl get pods -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,GPU:.spec.containers[0].resources.limits.'nvidia\.com/gpu'

# Inside pod — verify GPU access
kubectl exec -it gpu-pod -- nvidia-smi
kubectl exec -it gpu-pod -- python -c "import torch; print(torch.cuda.device_count())"

# Check DCGM exporter metrics
kubectl port-forward svc/nvidia-dcgm-exporter 9400:9400 -n gpu-operator
curl localhost:9400/metrics | grep DCGM_FI_DEV_GPU_UTIL
```

## Interview Questions

1. **Q: Training job ka GPU utilization 30% hai. Problem kya hai aur kaise fix karoge?**
   A: GPU idle hai = data loading slow hai (CPU/IO bottleneck). Fix: (1) num_workers increase in DataLoader, (2) pre-fetch data to local NVMe, (3) use FSx Lustre for fast parallel reads, (4) profile with nsight systems to find exact bottleneck.

2. **Q: XID 79 error aaya — "GPU has fallen off the bus". Kya karoge?**
   A: (1) Immediately cordon node, (2) Drain GPU pods, (3) Check dmesg for PCIe errors, (4) Try GPU reset (nvidia-smi -r), (5) If persists → hardware issue → reseat GPU physically or replace. Trigger node replacement if using HyperPod/managed cluster.

3. **Q: Cluster mein 100 GPUs hain. Kaise ensure karoge ki hardware failures early detect hon?**
   A: (1) DCGM exporter on all nodes, (2) Prometheus scraping every 15s, (3) Alerts on: ECC errors > 0, XID errors, temperature > 83°C, retired pages increasing, (4) Weekly dcgmi diag -r 3 during maintenance window, (5) Track retired pages trend — replace before failure.

4. **Q: Pod crash hua, nvidia-smi shows memory used but no process. Fix?**
   A: Zombie CUDA context. (1) Check `nvidia-smi` for orphaned processes, (2) `fuser -v /dev/nvidia*` to find hidden processes, (3) Kill them, (4) If persists → `nvidia-smi -r` to reset GPU, (5) Root cause: pod didn't cleanup CUDA context on exit (add preStop hook or use nvidia-container-toolkit properly).
