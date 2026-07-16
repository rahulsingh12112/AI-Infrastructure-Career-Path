# 06 — Multi-Model Endpoints & Multi-Container Endpoints

## The Problem

```
You have 100 ML models in production:
├── Each on its own endpoint = 100 × ml.g5.xlarge = $100/hr!
├── Most models get <1 request/minute (idle 99% of time)
├── Each endpoint needs minimum 1 instance (can't scale to zero)
└── Cost: $73,000/month for mostly idle GPUs!

Solution:
├── Multi-Model Endpoint: 100 models on 1 endpoint
├── Models loaded/unloaded dynamically from S3
├── Only active models in memory
└── Cost: 1-5 instances = $730-$3,650/month (95% savings!)
```

## Multi-Model Endpoint (MME)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Model Endpoint                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Client Request: POST /invocations                           │
│  Header: X-Amzn-SageMaker-Target-Model: model-A.tar.gz     │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────────────────────────────────────────────┐       │
│  │           Model Server (NVIDIA Triton)            │       │
│  │                                                   │       │
│  │  Memory:                                          │       │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐           │       │
│  │  │ Model A │ │ Model B │ │ Model C │ (in GPU)   │       │
│  │  └─────────┘ └─────────┘ └─────────┘           │       │
│  │                                                   │       │
│  │  S3 (model store):                               │       │
│  │  ├── model-A.tar.gz                              │       │
│  │  ├── model-B.tar.gz                              │       │
│  │  ├── model-C.tar.gz                              │       │
│  │  ├── ... (1000s of models)                       │       │
│  │  └── model-Z.tar.gz                              │       │
│  │                                                   │       │
│  │  Behavior:                                        │       │
│  │  1. Request for Model X arrives                   │       │
│  │  2. Model X in memory? → Serve immediately       │       │
│  │  3. Not in memory? → Load from S3 → Serve        │       │
│  │  4. Memory full? → Evict least-used model (LRU)  │       │
│  │                                                   │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Create Multi-Model Endpoint

```python
import sagemaker
from sagemaker.multidatamodel import MultiDataModel

session = sagemaker.Session()
role = sagemaker.get_execution_role()

# 1. Create MultiDataModel
multi_model = MultiDataModel(
    name="my-multi-model",
    model_data_prefix="s3://my-bucket/multi-model/models/",  # All models here
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1-gpu-py310",
    role=role,
    sagemaker_session=session,
)

# 2. Deploy endpoint
predictor = multi_model.deploy(
    initial_instance_count=2,
    instance_type="ml.g5.xlarge",
    endpoint_name="multi-model-endpoint",
)

# 3. Add models (upload to S3 prefix)
multi_model.add_model(
    model_data_source="s3://my-bucket/trained-models/fraud-v1/model.tar.gz",
    model_data_path="fraud-v1.tar.gz",
)
multi_model.add_model(
    model_data_source="s3://my-bucket/trained-models/fraud-v2/model.tar.gz",
    model_data_path="fraud-v2.tar.gz",
)

# 4. Invoke specific model
response = predictor.predict(
    data={"features": [1.0, 2.0, 3.0]},
    target_model="fraud-v1.tar.gz",  # Which model to use
)
```

### MME with GPU (NVIDIA Triton)

```python
from sagemaker.triton import TritonModel

# For GPU-based MME, use Triton Inference Server
model = TritonModel(
    model_data="s3://my-bucket/triton-models/",
    role=role,
    image_uri="301217895009.dkr.ecr.us-east-1.amazonaws.com/sagemaker-tritonserver:23.10-py3",
    sagemaker_session=session,
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.4xlarge",    # GPU instance
    endpoint_name="triton-mme-gpu",
)

# Triton model repository structure in S3:
# s3://my-bucket/triton-models/
# ├── model_a/
# │   ├── config.pbtxt
# │   └── 1/
# │       └── model.pt
# ├── model_b/
# │   ├── config.pbtxt
# │   └── 1/
# │       └── model.onnx
```

### MME Performance Considerations

```
Metric              │ Value           │ Impact
────────────────────┼─────────────────┼──────────────────────
Cold start (S3 load)│ 5-30 sec        │ First request to unloaded model
Hot inference       │ 5-50 ms         │ Model already in memory
Memory per model    │ 100MB - 5GB     │ Determines how many fit
Model eviction      │ LRU-based       │ Least-used models removed
Max models in memory│ GPU_MEM / model │ ml.g5.xlarge (24GB) ≈ 5-20 models

Optimization:
├── Use smaller model formats (ONNX, TensorRT) = fit more models
├── Quantize models (FP16, INT8) = 2-4x more models in memory
├── Warm up frequently-used models (pre-load)
├── Right-size instance (enough memory for hot models)
└── Use auto-scaling for burst traffic
```

## Multi-Container Endpoint (MCE)

### What's Different from MME?

```
MME: Multiple models, SAME container (same framework)
MCE: Multiple containers, different frameworks/models in pipeline

Use cases:
├── Pre-processing → Model A → Post-processing (serial pipeline)
├── Model A + Model B (ensemble, parallel)
├── Different frameworks (TensorFlow + PyTorch on same endpoint)
└── A/B testing (route % traffic to different containers)
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Container Endpoint                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SERIAL MODE (Pipeline):                                     │
│  Request → [Container 1: Preprocess] → [Container 2: Model] │
│         → [Container 3: Postprocess] → Response             │
│                                                              │
│  DIRECT MODE (A/B or Ensemble):                             │
│  Request → Router                                            │
│            ├── 80% → [Container 1: Model A]                 │
│            └── 20% → [Container 2: Model B]                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Create MCE (Serial Pipeline)

```python
from sagemaker.model import Model
from sagemaker.pipeline_model import PipelineModel

# Container 1: Feature engineering
preprocess_model = Model(
    image_uri="my-account.dkr.ecr.us-east-1.amazonaws.com/preprocessor:latest",
    model_data="s3://my-bucket/models/preprocessor/model.tar.gz",
    role=role,
)

# Container 2: ML Model (PyTorch)
ml_model = Model(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1-gpu-py310",
    model_data="s3://my-bucket/models/fraud-model/model.tar.gz",
    role=role,
)

# Container 3: Post-processing
postprocess_model = Model(
    image_uri="my-account.dkr.ecr.us-east-1.amazonaws.com/postprocessor:latest",
    model_data="s3://my-bucket/models/postprocessor/model.tar.gz",
    role=role,
)

# Pipeline: Request flows through all 3 containers in order
pipeline_model = PipelineModel(
    name="fraud-pipeline",
    role=role,
    models=[preprocess_model, ml_model, postprocess_model],
)

predictor = pipeline_model.deploy(
    initial_instance_count=2,
    instance_type="ml.g5.xlarge",
)
```

### MCE with Production Variants (A/B Testing)

```python
from sagemaker.model import Model

# Model A (current production)
model_a = Model(
    image_uri="pytorch-inference:2.1",
    model_data="s3://bucket/models/v1/model.tar.gz",
    role=role,
    name="model-a",
)

# Model B (challenger)
model_b = Model(
    image_uri="pytorch-inference:2.1",
    model_data="s3://bucket/models/v2/model.tar.gz",
    role=role,
    name="model-b",
)

# Deploy with traffic split
import boto3
sm = boto3.client("sagemaker")

sm.create_endpoint_config(
    EndpointConfigName="ab-test-config",
    ProductionVariants=[
        {
            "VariantName": "model-a",
            "ModelName": "model-a",
            "InstanceType": "ml.g5.xlarge",
            "InitialInstanceCount": 2,
            "InitialVariantWeight": 0.8,  # 80% traffic
        },
        {
            "VariantName": "model-b",
            "ModelName": "model-b",
            "InstanceType": "ml.g5.xlarge",
            "InitialInstanceCount": 1,
            "InitialVariantWeight": 0.2,  # 20% traffic
        },
    ],
)

# Gradually shift traffic
sm.update_endpoint_weights_and_capacities(
    EndpointName="my-endpoint",
    DesiredWeightsAndCapacities=[
        {"VariantName": "model-a", "DesiredWeight": 0.5},
        {"VariantName": "model-b", "DesiredWeight": 0.5},
    ],
)
```

## MME vs MCE vs Single-Model — Decision Matrix

```
Scenario                         │ Solution
─────────────────────────────────┼────────────────────────────
1 model, high traffic            │ Single-model endpoint
1 model, variable traffic        │ Single + auto-scaling
Many similar models, low traffic │ MME (Multi-Model)
Pipeline (preprocess→model→post) │ MCE Serial
A/B testing two models           │ MCE with variants
Ensemble (combine 2+ models)     │ MCE Direct or custom logic
Different frameworks             │ MCE (each container = framework)
100+ customer-specific models    │ MME (each customer = 1 model)
```

## Auto-Scaling for Endpoints

```python
# Auto-scaling based on invocations per instance
import boto3

aas = boto3.client("application-autoscaling")

# Register scalable target
aas.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/my-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=10,
)

# Scaling policy: Target tracking
aas.put_scaling_policy(
    PolicyName="gpu-utilization-tracking",
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/my-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 70.0,  # Target 70% GPU utilization
        "CustomizedMetricSpecification": {
            "MetricName": "GPUUtilization",
            "Namespace": "/aws/sagemaker/Endpoints",
            "Dimensions": [
                {"Name": "EndpointName", "Value": "my-endpoint"},
                {"Name": "VariantName", "Value": "AllTraffic"},
            ],
            "Statistic": "Average",
        },
        "ScaleInCooldown": 300,   # Wait 5 min before scale down
        "ScaleOutCooldown": 60,    # Scale up faster (1 min)
    },
)
```

## Interview Questions

1. **Q: 500 customer-specific models hain (each trained on customer data). Kaise deploy karoge cost-effectively?**
   A: Multi-Model Endpoint. All 500 models in S3 prefix. MME loads models on-demand (LRU cache). Top 20 customers (80% traffic) → models always in memory. Others loaded on demand (~10-30s cold start). Cost: 2-5 instances vs 500 instances. Add auto-scaling for peak hours.

2. **Q: MME pe cold start 30 sec hai. Customer complain kar raha hai. Fix?**
   A: (1) Pre-warm top models using scheduled invocations (keep them hot), (2) Use smaller model format (ONNX/TensorRT = faster load), (3) Use larger instance (more memory = more models cached), (4) Model compression (quantize to INT8 = 4x smaller = faster S3 download), (5) For critical models → dedicated single-model endpoint.

3. **Q: A/B test karna hai new model ka. Kaise set up karoge safely?**
   A: MCE with production variants: (1) Deploy v2 with 5% traffic weight, (2) Monitor metrics (latency, accuracy, error rate) for 7 days, (3) Gradually increase to 50% if metrics match/improve, (4) If regression → immediately set v2 weight to 0%, (5) If success → shift 100% to v2, remove v1.

4. **Q: Multi-container pipeline (preprocess→model→postprocess) pe latency high hai. Kaise debug?**
   A: (1) CloudWatch: Check per-container invocation time (which container is slow?), (2) If preprocess slow → optimize feature engineering code, (3) If model slow → profile GPU utilization, batch requests, (4) Container-to-container overhead → consider merging into single container, (5) Use smaller instances for preprocess/postprocess (don't need GPU).
