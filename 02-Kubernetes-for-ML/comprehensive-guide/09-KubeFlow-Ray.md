# 09 — KubeFlow & Ray on Kubernetes

## KubeFlow

### What is KubeFlow?
ML platform on Kubernetes. End-to-end ML lifecycle management.

```
KubeFlow Components:
├── Notebooks (Jupyter on K8s)
├── Pipelines (DAG workflows)
├── Training Operators (distributed training)
│   ├── PyTorchJob
│   ├── TFJob
│   ├── MPIJob
│   └── XGBoostJob
├── KServe (model serving)
├── Katib (hyperparameter tuning)
└── Volumes (data management)
```

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    KubeFlow Platform                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐         │
│  │ Notebook  │  │ Pipelines │  │   Katib      │         │
│  │ Server    │  │ (Argo)    │  │ (HP tuning)  │         │
│  └────┬─────┘  └─────┬─────┘  └──────┬───────┘         │
│       │               │               │                  │
│       ▼               ▼               ▼                  │
│  ┌──────────────────────────────────────────────┐       │
│  │           Training Operators                   │       │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐   │       │
│  │  │PyTorchJob │ │  TFJob    │ │  MPIJob   │   │       │
│  │  └───────────┘ └───────────┘ └───────────┘   │       │
│  └──────────────────────────────────────────────┘       │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────┐       │
│  │              KServe (Inference)                │       │
│  │  Model A ──→ AutoScaler ──→ GPU Pods          │       │
│  └──────────────────────────────────────────────┘       │
│                                                          │
│  All running on Kubernetes with GPU support              │
└─────────────────────────────────────────────────────────┘
```

### PyTorchJob (Most Important for Training)

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: llm-finetuning
  namespace: training
spec:
  nprocPerNode: "8"       # GPUs per node
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: nvcr.io/nvidia/pytorch:24.01-py3
            command:
            - torchrun
            - --nnodes=4
            - --nproc_per_node=8
            - --rdzv_backend=c10d
            - --rdzv_endpoint=$(MASTER_ADDR):$(MASTER_PORT)
            - train.py
            - --model=llama-7b
            - --data=/data/train
            - --checkpoint_dir=/checkpoints
            env:
            - name: NCCL_DEBUG
              value: "WARN"
            resources:
              limits:
                nvidia.com/gpu: 8
                vpc.amazonaws.com/efa: 4
            volumeMounts:
            - name: data
              mountPath: /data
            - name: checkpoints
              mountPath: /checkpoints
            - name: shm
              mountPath: /dev/shm
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: training-data-fsx
          - name: checkpoints
            persistentVolumeClaim:
              claimName: checkpoint-efs
          - name: shm
            emptyDir:
              medium: Memory
              sizeLimit: 64Gi
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: nvcr.io/nvidia/pytorch:24.01-py3
            command:
            - torchrun
            - --nnodes=4
            - --nproc_per_node=8
            - --rdzv_backend=c10d
            - --rdzv_endpoint=$(MASTER_ADDR):$(MASTER_PORT)
            - train.py
            resources:
              limits:
                nvidia.com/gpu: 8
                vpc.amazonaws.com/efa: 4
            volumeMounts:
            - name: data
              mountPath: /data
            - name: checkpoints
              mountPath: /checkpoints
            - name: shm
              mountPath: /dev/shm
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: training-data-fsx
          - name: checkpoints
            persistentVolumeClaim:
              claimName: checkpoint-efs
          - name: shm
            emptyDir:
              medium: Memory
              sizeLimit: 64Gi
```

### MPIJob (for Horovod-style training)

```yaml
apiVersion: kubeflow.org/v1
kind: MPIJob
metadata:
  name: horovod-training
spec:
  slotsPerWorker: 8
  runPolicy:
    cleanPodPolicy: Running
  mpiReplicaSpecs:
    Launcher:
      replicas: 1
      template:
        spec:
          containers:
          - name: mpi
            image: horovod/horovod:latest
            command:
            - mpirun
            - --allow-run-as-root
            - -np 32
            - -bind-to none
            - -x NCCL_DEBUG=INFO
            - python train.py
    Worker:
      replicas: 4
      template:
        spec:
          containers:
          - name: worker
            image: horovod/horovod:latest
            resources:
              limits:
                nvidia.com/gpu: 8
```

### KServe (Model Serving)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-serving
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      storageUri: s3://models/llama-7b/
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          cpu: "4"
          memory: "32Gi"
    minReplicas: 2
    maxReplicas: 10
    scaleTarget: 70        # Scale up at 70% GPU utilization
    scaleMetric: gpu       # Use GPU utilization for autoscaling
```

## Ray on Kubernetes

### What is Ray?
Distributed computing framework. Simpler than Horovod/MPI for many use cases.

```
Ray Components:
├── Ray Core (distributed execution)
├── Ray Train (distributed training)
├── Ray Serve (model serving)
├── Ray Tune (hyperparameter tuning)
└── Ray Data (data processing)
```

### KubeRay Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  KubeRay on Kubernetes                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  KubeRay Operator (manages Ray clusters)                 │
│  ├── Watches: RayCluster, RayJob, RayService CRDs       │
│  └── Creates/manages Ray head + worker pods              │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │              Ray Cluster                       │       │
│  │                                               │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │       │
│  │  │Ray Head │  │Worker 0 │  │Worker 1 │      │       │
│  │  │(no GPU) │  │(8 GPUs) │  │(8 GPUs) │      │       │
│  │  │Dashboard│  │         │  │         │      │       │
│  │  │GCS      │  │         │  │         │      │       │
│  │  └─────────┘  └─────────┘  └─────────┘      │       │
│  │                                               │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### RayCluster for Training

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: gpu-training-cluster
spec:
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
              cpu: "8"
              memory: "32Gi"
            requests:
              cpu: "4"
              memory: "16Gi"
  workerGroupSpecs:
  - replicas: 4
    groupName: gpu-workers
    rayStartParams:
      num-gpus: "8"
    template:
      spec:
        containers:
        - name: ray-worker
          image: rayproject/ray-ml:2.9.0-py310-gpu
          resources:
            limits:
              nvidia.com/gpu: 8
              cpu: "96"
              memory: "768Gi"
            requests:
              nvidia.com/gpu: 8
              cpu: "48"
              memory: "384Gi"
          volumeMounts:
          - name: shm
            mountPath: /dev/shm
        volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 64Gi
```

### RayJob (Submit Training Job)

```yaml
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: finetune-llama
spec:
  entrypoint: "python train.py --model llama-7b --num_workers 4"
  runtimeEnvYAML: |
    pip:
      - transformers
      - accelerate
      - datasets
    env_vars:
      NCCL_DEBUG: "WARN"
  rayClusterSpec:
    headGroupSpec:
      rayStartParams: {}
      template:
        spec:
          containers:
          - name: ray-head
            image: my-training:latest
            resources:
              limits:
                cpu: "4"
                memory: "16Gi"
    workerGroupSpecs:
    - replicas: 4
      groupName: gpu-workers
      rayStartParams:
        num-gpus: "8"
      template:
        spec:
          containers:
          - name: ray-worker
            image: my-training:latest
            resources:
              limits:
                nvidia.com/gpu: 8
```

### Ray Serve (Model Serving with Autoscaling)

```python
# Ray Serve deployment
from ray import serve
from ray.serve.handle import DeploymentHandle
import torch

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 10,
        "target_ongoing_requests": 5,
    },
)
class LLMServing:
    def __init__(self):
        self.model = torch.load("/models/llama-7b.pt")
        self.model.cuda()
        self.model.eval()
    
    async def __call__(self, request):
        data = await request.json()
        with torch.no_grad():
            output = self.model.generate(data["prompt"])
        return {"response": output}

# Deploy
app = LLMServing.bind()
serve.run(app, host="0.0.0.0", port=8000)
```

### RayService on K8s

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: llm-service
spec:
  serviceUnhealthySecondThreshold: 300
  deploymentUnhealthySecondThreshold: 300
  serveConfigV2: |
    applications:
    - name: llm_app
      route_prefix: /
      import_path: serve_llm:app
      deployments:
      - name: LLMServing
        num_replicas: 2
        ray_actor_options:
          num_gpus: 1
        autoscaling_config:
          min_replicas: 2
          max_replicas: 10
  rayClusterConfig:
    headGroupSpec:
      template:
        spec:
          containers:
          - name: ray-head
            image: my-serving:latest
    workerGroupSpecs:
    - replicas: 2
      groupName: gpu-workers
      template:
        spec:
          containers:
          - name: ray-worker
            image: my-serving:latest
            resources:
              limits:
                nvidia.com/gpu: 1
```

## KubeFlow vs Ray — When to Use What

```
Feature          │ KubeFlow                    │ Ray
─────────────────┼─────────────────────────────┼──────────────────────
Training         │ PyTorchJob (K8s native)     │ Ray Train (Ray native)
Serving          │ KServe (K8s native)         │ Ray Serve (simpler)
Pipelines        │ Yes (Argo-based)            │ No (use with Airflow)
HP Tuning        │ Katib                       │ Ray Tune
Notebooks        │ Built-in                    │ No
Complexity       │ Heavy (many components)     │ Lighter
GPU scheduling   │ Via K8s/Volcano             │ Ray handles internally
Best for         │ Enterprise ML platform      │ Distributed compute
Community        │ Large, CNCF                 │ Large, Anyscale
```

## Interview Questions

1. **Q: PyTorchJob vs MPIJob — kab kya use karoge?**
   A: PyTorchJob for torchrun-based training (newer, built-in fault tolerance, elastic). MPIJob for Horovod-based legacy training or when you need MPI collectives. New projects should use PyTorchJob with torchrun.

2. **Q: Ray Serve vs KServe — difference?**
   A: KServe = K8s native, supports multiple frameworks (TF, PyTorch, ONNX), built-in canary/A-B testing, uses KPA for autoscaling. Ray Serve = simpler Python-native, better for complex inference pipelines (chaining models), dynamic batching built-in.

3. **Q: 100-node training cluster pe KubeFlow set up karna hai. Key considerations?**
   A: (1) Separate GPU node pools for training vs inference, (2) NVIDIA GPU Operator + device plugin, (3) FSx Lustre for shared storage, (4) Volcano scheduler for gang scheduling, (5) EFA enabled for NCCL, (6) RBAC + namespaces per team, (7) Resource quotas per namespace.

4. **Q: Ray autoscaling kaise kaam karta hai K8s pe?**
   A: KubeRay operator watches Ray cluster resource demands → when Ray needs more workers → operator creates new K8s pods → Karpenter provisions new GPU nodes → Ray workers join cluster. Scale down: idle workers → Ray removes → KubeRay deletes pods → Karpenter terminates nodes.
