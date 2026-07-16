# 05 — Model Registry & Lineage Tracking

## What is Model Registry?

Central catalog for all your trained models — versioned, with metadata, approval workflows.

```
Without Model Registry:
├── Models saved in random S3 paths
├── "Which model is in production?" → Nobody knows
├── "What data was this trained on?" → Lost information
├── Rolling back? → Which version was previous?
└── Audit compliance? → Impossible

With Model Registry:
├── Every model version registered with metadata
├── Approval workflow (Pending → Approved → Deployed)
├── Full lineage (data → training job → model → endpoint)
├── Easy rollback (deploy previous version in 1 click)
└── Audit trail (who approved, when, why)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SageMaker Model Registry                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Model Package Group: "fraud-detection-model"                │
│  ├── Version 1 (Approved, deployed to prod)                 │
│  │   ├── Artifacts: s3://bucket/models/v1/model.tar.gz      │
│  │   ├── Metrics: accuracy=0.94, f1=0.91                    │
│  │   ├── Training Job: fraud-train-2024-01-15               │
│  │   ├── Data: s3://bucket/data/train-v1/                   │
│  │   ├── Container: 123456.dkr.ecr/fraud-model:v1          │
│  │   └── Status: Approved (by: admin@company.com)           │
│  │                                                           │
│  ├── Version 2 (PendingManualApproval)                      │
│  │   ├── Artifacts: s3://bucket/models/v2/model.tar.gz      │
│  │   ├── Metrics: accuracy=0.96, f1=0.93                    │
│  │   ├── Training Job: fraud-train-2024-02-01               │
│  │   └── Status: PendingManualApproval                      │
│  │                                                           │
│  └── Version 3 (Rejected)                                    │
│      ├── Metrics: accuracy=0.88 (below threshold!)          │
│      └── Status: Rejected (reason: accuracy drop)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Model Registry Operations

### Create Model Package Group

```python
import boto3

sm_client = boto3.client("sagemaker")

# Create group (like a "repository" for model versions)
sm_client.create_model_package_group(
    ModelPackageGroupName="fraud-detection-model",
    ModelPackageGroupDescription="Fraud detection models for payment processing",
    Tags=[
        {"Key": "Team", "Value": "ML-Platform"},
        {"Key": "Domain", "Value": "Payments"},
    ],
)
```

### Register a Model Version

```python
from sagemaker.model import Model
from sagemaker import ModelPackage

# After training, register model
model_package = sm_client.create_model_package(
    ModelPackageGroupName="fraud-detection-model",
    ModelPackageDescription="v2: Retrained with Jan 2024 data",
    
    InferenceSpecification={
        "Containers": [
            {
                "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1-gpu-py310",
                "ModelDataUrl": "s3://my-bucket/models/v2/model.tar.gz",
                "Framework": "PYTORCH",
            }
        ],
        "SupportedTransformInstanceTypes": ["ml.g5.xlarge"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.g5.xlarge", "ml.p4d.24xlarge"],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    
    # Model metrics
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": "s3://my-bucket/evaluation/metrics.json",
            },
        },
        "Bias": {
            "Report": {
                "ContentType": "application/json",
                "S3Uri": "s3://my-bucket/evaluation/bias-report.json",
            },
        },
    },
    
    # Approval status
    ModelApprovalStatus="PendingManualApproval",
    
    # Custom metadata
    CustomerMetadataProperties={
        "training_data_version": "2024-01-15",
        "training_job": "fraud-train-20240201",
        "feature_count": "128",
        "algorithm": "transformer",
    },
)

model_package_arn = model_package["ModelPackageArn"]
print(f"Registered: {model_package_arn}")
```

### Approve/Reject Model

```python
# Approve for production deployment
sm_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Approved",
    ApprovalDescription="Accuracy improved 2% over v1. A/B tested for 7 days.",
)

# Reject
sm_client.update_model_package(
    ModelPackageArn=model_package_arn,
    ModelApprovalStatus="Rejected",
    ApprovalDescription="Accuracy below threshold (0.88 < 0.90 required).",
)
```

### Deploy from Registry

```python
from sagemaker import ModelPackage

# Deploy latest approved version
model = ModelPackage(
    role=role,
    model_package_arn=model_package_arn,
    sagemaker_session=session,
)

predictor = model.deploy(
    initial_instance_count=2,
    instance_type="ml.g5.xlarge",
    endpoint_name="fraud-detection-prod",
)
```

## Lineage Tracking

### What is Lineage?

Track the complete history: Data → Processing → Training → Model → Endpoint

```
┌──────────────────────────────────────────────────────────┐
│                    Lineage Graph                           │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  DataSet (S3)                                             │
│  s3://data/raw/transactions-2024.csv                      │
│       │                                                   │
│       ▼ [Input]                                           │
│  ProcessingJob: "preprocess-20240201"                     │
│       │                                                   │
│       ▼ [Output]                                          │
│  DataSet (S3)                                             │
│  s3://data/processed/train.parquet                        │
│       │                                                   │
│       ▼ [Input]                                           │
│  TrainingJob: "fraud-train-20240201"                      │
│  (hyperparams: lr=5e-5, epochs=10, batch=64)             │
│       │                                                   │
│       ▼ [Output]                                          │
│  Model: "fraud-detection-v2"                              │
│  s3://models/v2/model.tar.gz                             │
│       │                                                   │
│       ▼ [Deployed]                                        │
│  Endpoint: "fraud-detection-prod"                         │
│  (instance: ml.g5.xlarge, count: 2)                      │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Query Lineage

```python
from sagemaker.lineage.context import Context
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association

# Find all artifacts related to a model
model_artifact = Artifact.list(
    source_uri="s3://my-bucket/models/v2/model.tar.gz",
    sagemaker_session=session,
)

# Find upstream (what data/jobs created this model)
associations = Association.list(
    destination_arn=model_artifact_arn,
    sagemaker_session=session,
)
# Returns: Processing job, Training job, Dataset artifacts

# Find downstream (where is this model deployed)
downstream = Association.list(
    source_arn=model_artifact_arn,
    sagemaker_session=session,
)
# Returns: Endpoints using this model

# Track data → model (full chain)
# "Which models were trained on this specific dataset?"
dataset_artifact = Artifact.list(
    source_uri="s3://data/raw/transactions-2024.csv",
    sagemaker_session=session,
)
# Follow associations downstream to find all models
```

### Lineage in Pipelines (Automatic)

```python
# SageMaker Pipelines AUTOMATICALLY tracks lineage:
# - Every ProcessingStep input/output → Artifact
# - Every TrainingStep → links training job to model artifact
# - Every RegisterModel → links model to model package
# - Every DeployStep → links model to endpoint

# You don't need to write lineage code!
# Just use SageMaker Pipeline steps and lineage is tracked.
```

## Model Versioning Strategy

```
Versioning Best Practices:
├── Semantic versioning: Major.Minor.Patch
│   ├── Major: Architecture change (BERT → GPT)
│   ├── Minor: Retrained with new data
│   └── Patch: Hyperparameter tweak
│
├── Approval workflow:
│   ├── Auto-approve: If metrics > threshold (dev/staging)
│   └── Manual approve: Production deployments
│
├── Rollback strategy:
│   ├── Keep last 5 approved versions deployable
│   ├── Blue-green deployment for production
│   └── Canary: Route 5% traffic → new model, monitor
│
└── Retention policy:
    ├── Approved models: Keep forever
    ├── Rejected models: Delete after 30 days
    └── Pending models: Auto-reject after 7 days
```

## Integration with CI/CD

```python
# Automated pipeline: Train → Evaluate → Register → Approve → Deploy

# In SageMaker Pipeline:
# 1. Train model
# 2. Evaluate metrics
# 3. If metrics > threshold:
#    - Register in Model Registry (PendingManualApproval)
#    - Send Slack notification to ML team
# 4. Team reviews → Approves in console/API
# 5. EventBridge detects approval → triggers deployment pipeline
# 6. Deploy to staging → run integration tests → deploy to production

# EventBridge rule for auto-deployment on approval:
{
    "source": ["aws.sagemaker"],
    "detail-type": ["SageMaker Model Package State Change"],
    "detail": {
        "ModelPackageGroupName": ["fraud-detection-model"],
        "ModelApprovalStatus": ["Approved"]
    }
}
# → Triggers Lambda/CodePipeline → deploys to production endpoint
```

## Interview Questions

1. **Q: Model Registry vs S3 mein model save karna — kya fark hai?**
   A: S3 = just file storage (no versioning, no metadata, no approval). Registry = versioned catalog with metadata (metrics, lineage, training job link), approval workflow, deployment specification, and audit trail. Production-grade ML systems NEED a registry for governance, rollback, and compliance.

2. **Q: Production model mein bug mila. Kaise rollback karoge?**
   A: (1) Model Registry mein previous approved version find karo, (2) `model.deploy()` with previous version's ARN, (3) Update endpoint to point to old model (blue-green), (4) Mark current version as "Rejected" with reason, (5) Investigate and retrain. Total rollback time: ~5 min (just switching model version on endpoint).

3. **Q: Lineage tracking kab useful hota hai?**
   A: (1) Model drift detected → trace back to find which training data caused it, (2) Data quality issue → find ALL models trained on that data (impact analysis), (3) Audit/compliance → prove model was trained on approved data, (4) Debugging → reproduce exact training conditions, (5) Cost attribution → which data pipeline produced which model.

4. **Q: 50 ML models hain production mein. Registry kaise organize karoge?**
   A: (1) One ModelPackageGroup per model (fraud-detection, recommendation, etc.), (2) Tags for team/domain/criticality, (3) Naming convention: `{domain}-{model}-{version}`, (4) Approval policies: Tier-1 (revenue-critical) = manual + A/B test, Tier-2 = auto if metrics pass, (5) Retention: Auto-archive rejected versions after 30 days.
