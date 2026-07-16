# 01 — SageMaker Pipelines (ML CI/CD)

## What is SageMaker Pipelines?

ML CI/CD system — automates the entire ML lifecycle as a DAG (Directed Acyclic Graph).

```
Traditional ML:
├── Data scientist runs notebook manually
├── Copy-paste code to production
├── No versioning, no reproducibility
├── "It works on my machine" problem
└── No automated retraining

With Pipelines:
├── Define ML workflow as code (DAG)
├── Automated: data prep → train → evaluate → deploy
├── Versioned, reproducible, auditable
├── Scheduled retraining (daily/weekly)
└── Approval gates before production deployment
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   SageMaker Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐             │
│  │  Data     │───→│ Training │───→│ Evaluation│             │
│  │Processing │    │  Step    │    │   Step    │             │
│  └──────────┘    └──────────┘    └─────┬─────┘             │
│                                        │                     │
│                                   ┌────▼────┐                │
│                                   │Condition│                │
│                                   │accuracy │                │
│                                   │> 0.95?  │                │
│                                   └────┬────┘                │
│                              YES ┌─────┴─────┐ NO            │
│                                  ▼           ▼               │
│                           ┌───────────┐  ┌────────┐         │
│                           │Register   │  │  Fail  │         │
│                           │Model      │  │  Step  │         │
│                           └─────┬─────┘  └────────┘         │
│                                 │                            │
│                                 ▼                            │
│                           ┌───────────┐                      │
│                           │  Deploy   │                      │
│                           │ Endpoint  │                      │
│                           └───────────┘                      │
│                                                              │
│  Metadata Store:                                             │
│  ├── Pipeline execution history                              │
│  ├── Step inputs/outputs (S3 URIs)                          │
│  ├── Parameters used                                         │
│  ├── Metrics (accuracy, loss)                               │
│  └── Lineage tracking                                        │
└─────────────────────────────────────────────────────────────┘
```

## Pipeline Steps (All Types)

```
Step Type           │ What it does                         │ Compute
────────────────────┼──────────────────────────────────────┼──────────────
ProcessingStep      │ Data preprocessing, feature eng      │ Processing job
TrainingStep        │ Model training                       │ Training job
TransformStep       │ Batch inference                      │ Transform job
CreateModelStep     │ Create deployable model              │ API call
RegisterModel       │ Register in Model Registry           │ API call
ConditionStep       │ If/else branching                    │ None (logic)
CallbackStep        │ Wait for external approval           │ None (waits)
LambdaStep          │ Run AWS Lambda function              │ Lambda
QualityCheckStep    │ Data/model quality checks            │ Processing job
ClarifyCheckStep    │ Bias & explainability checks         │ Processing job
FailStep            │ Mark pipeline as failed              │ None
TuningStep          │ Hyperparameter tuning                │ HPO job
AutoMLStep          │ AutoML (Autopilot)                   │ AutoML job
```

## Complete Pipeline Code

```python
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.processing import ScriptProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

session = sagemaker.Session()
role = sagemaker.get_execution_role()

# ============ PIPELINE PARAMETERS ============
input_data = ParameterString(name="InputData", default_value="s3://my-bucket/data/")
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.95)
instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.p4d.24xlarge")

# ============ STEP 1: DATA PROCESSING ============
processor = ScriptProcessor(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1-gpu-py310",
    role=role,
    instance_count=2,
    instance_type="ml.m5.4xlarge",
    command=["python3"],
)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=processor,
    code="scripts/preprocess.py",
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input",
        )
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/output/train",
        ),
        sagemaker.processing.ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/output/test",
        ),
    ],
)

# ============ STEP 2: TRAINING ============
estimator = Estimator(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1-gpu-py310",
    role=role,
    instance_count=4,
    instance_type=instance_type,
    volume_size=500,
    max_run=86400,           # 24 hours max
    output_path="s3://my-bucket/models/",
    hyperparameters={
        "epochs": 10,
        "batch_size": 64,
        "learning_rate": 0.001,
    },
    # Distributed training
    distribution={
        "torch_distributed": {
            "enabled": True,
        }
    },
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="application/x-npy",
        ),
        "test": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            content_type="application/x-npy",
        ),
    },
)

# ============ STEP 3: EVALUATION ============
eval_processor = ScriptProcessor(
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1-gpu-py310",
    role=role,
    instance_count=1,
    instance_type="ml.g5.xlarge",
    command=["python3"],
)

eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=eval_processor,
    code="scripts/evaluate.py",
    inputs=[
        sagemaker.processing.ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        sagemaker.processing.ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test",
        ),
    ],
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
        ),
    ],
    property_files=[
        sagemaker.workflow.properties.PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation.json",
        ),
    ],
)

# ============ STEP 4: CONDITION (accuracy check) ============
from sagemaker.workflow.functions import JsonGet

accuracy_condition = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=eval_step.name,
        property_file="EvaluationReport",
        json_path="metrics.accuracy",
    ),
    right=accuracy_threshold,
)

# ============ STEP 5: REGISTER MODEL ============
register_step = RegisterModel(
    name="RegisterModel",
    estimator=estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.g5.xlarge", "ml.p4d.24xlarge"],
    transform_instances=["ml.g5.xlarge"],
    model_package_group_name="MyModelPackageGroup",
    approval_status=model_approval_status,
)

# ============ STEP 6: CONDITION STEP ============
from sagemaker.workflow.fail_step import FailStep

fail_step = FailStep(
    name="AccuracyTooLow",
    error_message="Model accuracy below threshold. Training failed.",
)

condition_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[accuracy_condition],
    if_steps=[register_step],     # If accuracy >= threshold → register
    else_steps=[fail_step],       # Else → fail
)

# ============ BUILD PIPELINE ============
pipeline = Pipeline(
    name="MLTrainingPipeline",
    parameters=[input_data, model_approval_status, accuracy_threshold, instance_type],
    steps=[processing_step, training_step, eval_step, condition_step],
)

# Create/Update pipeline
pipeline.upsert(role_arn=role)

# Execute pipeline
execution = pipeline.start(
    parameters={
        "InputData": "s3://my-bucket/data/v2/",
        "TrainingInstanceType": "ml.p4d.24xlarge",
    }
)

# Monitor
execution.describe()
execution.list_steps()
```

## Pipeline Scheduling & Triggers

```python
# Schedule: Run daily at 2 AM UTC
import json

pipeline_schedule = {
    "PipelineName": "MLTrainingPipeline",
    "ScheduleExpression": "cron(0 2 * * ? *)",   # Daily 2 AM
    "PipelineParameters": [
        {"Name": "InputData", "Value": "s3://my-bucket/data/latest/"},
    ],
}

# Trigger: Run on S3 data change (via EventBridge)
# EventBridge Rule → triggers pipeline when new data lands in S3

# Trigger: Run on model drift detection
# SageMaker Model Monitor → detects drift → EventBridge → Pipeline start
```

## Pipeline Best Practices

```
1. Parameterize everything:
   ├── Instance types (can change without code change)
   ├── Data paths
   ├── Hyperparameters
   └── Thresholds

2. Caching:
   ├── SageMaker caches step outputs
   ├── If inputs haven't changed → skip step
   ├── Saves time and cost on reruns
   └── Enable: step.add_depends_on() or automatic via SageMaker

3. Error handling:
   ├── FailStep for known failure conditions
   ├── RetryPolicy for transient errors
   ├── CallbackStep for manual approval
   └── SNS notifications on failure

4. Cost optimization:
   ├── Use spot instances for training step
   ├── Smallest instance for processing
   ├── Terminate early if metrics plateau
   └── Cache intermediate artifacts
```

## CI/CD Integration (GitOps)

```yaml
# CodePipeline/GitHub Actions for Pipeline-as-Code:

# .github/workflows/ml-pipeline.yml
name: ML Pipeline CI/CD
on:
  push:
    branches: [main]
    paths:
      - 'pipelines/**'
      - 'scripts/**'

jobs:
  deploy-pipeline:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        role-to-assume: arn:aws:iam::123456789:role/github-actions
        aws-region: us-east-1
    
    - name: Update SageMaker Pipeline
      run: |
        pip install sagemaker boto3
        python pipelines/create_pipeline.py --update
    
    - name: Run Pipeline (staging)
      run: |
        python pipelines/run_pipeline.py --env staging
    
    - name: Wait for completion
      run: |
        python pipelines/wait_pipeline.py --timeout 7200
```

## Interview Questions

1. **Q: SageMaker Pipeline vs Airflow vs Step Functions — kab kya use karoge?**
   A: SageMaker Pipelines: ML-specific, integrated with SageMaker (model registry, endpoints, training jobs). Best when already using SageMaker. Airflow: General-purpose orchestration, more flexible, better for complex non-ML workflows mixed with ML. Step Functions: AWS-native, good for simple workflows, serverless. For ML-focused teams → SageMaker Pipelines. For platform teams with mixed workloads → Airflow.

2. **Q: Pipeline mein training step fail ho gaya spot interruption se. Kaise handle karoge?**
   A: (1) TrainingStep mein RetryPolicy set karo (max_retries=3), (2) Estimator mein use_spot_instances=True with max_wait, (3) Checkpointing enabled in training script, (4) SageMaker automatically resumes from checkpoint on spot retry.

3. **Q: Model retrain kab trigger karna chahiye?**
   A: (1) Scheduled (weekly/daily for data freshness), (2) Data drift detected (Model Monitor → EventBridge → Pipeline), (3) Model performance degraded (accuracy drops below threshold), (4) New data volume crosses threshold (enough new samples), (5) Code change (CI/CD push → pipeline update + run).

4. **Q: Pipeline execution 6 hours le raha hai. Kaise optimize karoge?**
   A: (1) Enable caching (unchanged steps skip), (2) Parallel processing steps where possible, (3) Spot instances for training, (4) Right-size instances per step (processing doesn't need GPU), (5) Pre-built container images (avoid pip install at runtime), (6) Reduce data scope for development pipeline.
