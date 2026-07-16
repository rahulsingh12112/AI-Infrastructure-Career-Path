# 02 — Self-hosted LLMs on EKS (vLLM + GPU Nodes)

## Why Self-host?

```
Self-hosted = Run your own LLM on your own GPU infrastructure

When it makes sense:
├── High volume: >$20K/month on Bedrock → self-host is cheaper
├── Custom models: Fine-tuned models not available on Bedrock
├── Low latency: No API gateway hop (direct GPU access)
├── Data sovereignty: Model + data never leave your VPC
├── Full control: Custom batching, caching, routing logic
└── Open-weight models: LLaMA, Mistral, Mixtral (free to use)
```

## Production Architecture on EKS

```
┌─────────────────────────────────────────────────────────────┐
│           Self-hosted LLM on EKS (Production)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Clients (Apps, APIs)                                        │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────────────────────────────────────┐       │
│  │  ALB / Istio Gateway                              │       │
│  │  ├── TLS termination                             │       │
│  │  ├── Rate limiting                               │       │
│  │  ├── Request routing (model A vs B)              │       │
│  │  └── Health checks                               │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│                         ▼                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │  vLLM Pods (GPU)                                  │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐         │       │
│  │  │ vLLM-0   │ │ vLLM-1   │ │ vLLM-2   │ (HPA)  │       │
│  │  │ LLaMA-70B│ │ LLaMA-70B│ │ LLaMA-70B│         │       │
│  │  │ 4×A100   │ │ 4×A100   │ │ 4×A100   │         │       │
│  │  └──────────┘ └──────────┘ └──────────┘         │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  Infrastructure:                                             │
│  ├── Karpenter (auto-provision GPU nodes)                   │
│  ├── NVIDIA GPU Operator (drivers, device plugin)           │
│  ├── EFA (high-speed networking for TP)                     │
│  ├── FSx/EFS (model weights shared storage)                 │
│  └── Prometheus + Grafana (monitoring)                       │
│                                                              │
│  Storage:                                                    │
│  ├── Model weights: S3 → FSx (cached) or EFS              │
│  ├── Served via PVC mount into vLLM pods                   │
│  └── Multiple pods share same model files (ReadWriteMany)  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Complete Deployment (Step by Step)

### Step 1: EKS Cluster with GPU Support

```bash
# Create cluster
eksctl create cluster \
  --name llm-cluster \
  --region us-east-1 \
  --version 1.29 \
  --without-nodegroup

# Install GPU Operator
helm install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace

# Install Karpenter
helm install karpenter oci://public.ecr.aws/karpenter/karpenter -n karpenter --create-namespace
```

### Step 2: Karpenter NodePool for GPU

```yaml
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: llm-inference
spec:
  template:
    metadata:
      labels:
        workload: llm-inference
    spec:
      requirements:
      - key: "node.kubernetes.io/instance-type"
        operator: In
        values: ["g5.12xlarge", "g5.48xlarge", "p4d.24xlarge"]
      - key: "karpenter.sh/capacity-type"
        operator: In
        values: ["on-demand"]
      taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
      nodeClassRef:
        name: gpu-node-class
  limits:
    nvidia.com/gpu: 32
  disruption:
    consolidationPolicy: WhenUnderutilized
    consolidateAfter: 10m

---
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: gpu-node-class
spec:
  amiFamily: AL2
  instanceProfile: "karpenter-gpu-profile"
  blockDeviceMappings:
  - deviceName: /dev/xvda
    ebs:
      volumeSize: 200Gi
      volumeType: gp3
```

### Step 3: Model Storage (EFS for shared access)

```yaml
# EFS StorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: fs-xxxxx
  directoryPerms: "700"

---
# PVC for model weights
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-weights
spec:
  accessModes: [ReadWriteMany]
  storageClassName: efs-sc
  resources:
    requests:
      storage: 200Gi
```

### Step 4: Download Model (Init Job)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: download-model
spec:
  template:
    spec:
      containers:
      - name: downloader
        image: python:3.10
        command:
        - bash
        - -c
        - |
          pip install huggingface_hub
          huggingface-cli download meta-llama/Llama-2-70b-chat-hf \
            --local-dir /models/llama-70b \
            --token $HF_TOKEN
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-secret
              key: token
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-weights
      restartPolicy: Never
```

### Step 5: vLLM Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-70b
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-llama
  template:
    metadata:
      labels:
        app: vllm-llama
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "/models/llama-70b"
        - "--tensor-parallel-size"
        - "4"
        - "--gpu-memory-utilization"
        - "0.92"
        - "--max-model-len"
        - "4096"
        - "--max-num-seqs"
        - "128"
        - "--enable-prefix-caching"
        - "--dtype"
        - "bfloat16"
        ports:
        - containerPort: 8000
          name: http
        resources:
          limits:
            nvidia.com/gpu: 4
          requests:
            cpu: "16"
            memory: "64Gi"
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: shm
          mountPath: /dev/shm
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300  # Model loading takes time
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 600
          periodSeconds: 30
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-weights
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 16Gi

---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm-llama
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP

---
# Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-ingress
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internal
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /health
spec:
  rules:
  - http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-service
            port:
              number: 8000
```

### Step 6: Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama-70b
  minReplicas: 2
  maxReplicas: 8
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 1
        periodSeconds: 120     # 1 pod every 2 min (GPU node spin-up time)
    scaleDown:
      stabilizationWindowSeconds: 600  # 10 min before scale down
  metrics:
  - type: Pods
    pods:
      metric:
        name: vllm_num_requests_waiting
      target:
        type: AverageValue
        averageValue: "10"
```

### Step 7: Monitoring

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm-monitor
spec:
  selector:
    matchLabels:
      app: vllm-llama
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
```

## Production Checklist

```
Pre-deployment:
├── [ ] Model weights downloaded and validated (checksum)
├── [ ] GPU quota sufficient in AWS account
├── [ ] EFA enabled (if tensor parallel across nodes)
├── [ ] Security groups allow pod-to-pod traffic
├── [ ] Model license compliance checked (LLaMA license)

Deployment:
├── [ ] Readiness probe configured (wait for model load)
├── [ ] Resource limits set (GPU, CPU, memory)
├── [ ] Shared memory (/dev/shm) mounted and sized
├── [ ] PDB configured (don't kill pods mid-generation)
├── [ ] Node affinity/taints for GPU isolation

Operations:
├── [ ] HPA configured with appropriate metric
├── [ ] Monitoring dashboard (TTFT, TPOT, throughput, errors)
├── [ ] Alerting (high latency, errors, GPU failures)
├── [ ] Log aggregation (CloudWatch/Fluentd)
├── [ ] Model update strategy (blue/green or rolling)

Security:
├── [ ] VPC-only access (no public endpoint)
├── [ ] IAM roles for service account (IRSA)
├── [ ] Network policies (restrict pod communication)
├── [ ] Secrets management (HF token in Secrets Manager)
├── [ ] API authentication (JWT/API keys)
```

## Interview Questions

1. **Q: EKS pe LLaMA-70B deploy karna hai production-grade. Architecture kya hogi?**
   A: (1) p4d.24xlarge nodes (8×A100), TP=4 for 70B. (2) vLLM with prefix caching, BF16. (3) Model on EFS (ReadWriteMany, shared across pods). (4) 2 replicas minimum (HA). (5) ALB ingress (internal). (6) HPA on queue depth. (7) Karpenter for GPU auto-provisioning. (8) NVIDIA GPU Operator. (9) Prometheus + Grafana monitoring. (10) PDB to prevent disruption during serving.

2. **Q: Model loading mein 5 min lag rahe hain. Cold start kaise reduce karoge?**
   A: (1) Use local NVMe instead of EFS (faster reads, 7 GB/s vs 1 GB/s), (2) Keep minimum replicas always warm (minReplicas=2), (3) Pre-pull container image (large ~20GB images), (4) Use safetensors format (faster than pickle), (5) Predictive scaling (scale before peak), (6) Smaller quantized model (AWQ 4-bit loads faster, less data to read).

3. **Q: Single vLLM pod crash hua mid-serving. Impact kya hoga?**
   A: (1) Active requests on that pod = lost (clients get error), (2) Load balancer detects via failed health check (~30s), (3) Routes traffic to remaining pods, (4) HPA may scale up replacement, (5) Impact: Brief latency spike on remaining pods. Mitigation: (a) PDB ensures min availability, (b) Client-side retry with exponential backoff, (c) Run 2+ replicas always.

4. **Q: EFS pe model load slow hai. Alternative?**
   A: Options: (1) FSx for Lustre (10-50x faster parallel reads), (2) Local NVMe (instance store, fastest but lost on termination), (3) EBS snapshot → mount at boot (fast, per-node), (4) S3 + Mountpoint (good for read-once pattern), (5) Best: Init container downloads to local NVMe, vLLM reads from local. Trade-off: NVMe = fastest but model copy per node.
