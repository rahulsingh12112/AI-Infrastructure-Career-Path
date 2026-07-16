# 08 — Hands-On Labs (SageMaker Advanced)

## Prerequisites

```bash
# Setup
pip install sagemaker boto3 torch transformers datasets
aws configure  # Set up credentials

# Verify
python -c "import sagemaker; print(sagemaker.__version__)"
aws sts get-caller-identity
```

---

## Lab 1: Build an End-to-End ML Pipeline

### Objective
Create a SageMaker Pipeline that preprocesses data, trains a model, evaluates it, and registers it.

### Steps

```python
# pipeline_lab.py
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.parameters import ParameterString
from sagemaker.processing import ScriptProcessor
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Parameters
input_data = ParameterString(name="InputData", default_value="s3://sagemaker-sample-data-us-east-1/processing/census/")

# Step 1: Preprocessing
processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", session.boto_region_name, "1.2-1"),
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    command=["python3"],
)

preprocess_step = ProcessingStep(
    name="Preprocess",
    processor=processor,
    code="preprocess.py",  # Your preprocessing script
    inputs=[sagemaker.processing.ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        sagemaker.processing.ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        sagemaker.processing.ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
    ],
)

# Step 2: Training
estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_count=1,
    instance_type="ml.g5.xlarge",
    framework_version="2.1",
    py_version="py310",
    hyperparameters={"epochs": 5, "batch_size": 32},
)

train_step = TrainingStep(
    name="Train",
    estimator=estimator,
    inputs={"train": sagemaker.inputs.TrainingInput(
        s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri
    )},
)

# Step 3: Register Model
register_step = RegisterModel(
    name="Register",
    estimator=estimator,
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.g5.xlarge"],
    model_package_group_name="lab-model-group",
    approval_status="PendingManualApproval",
)

# Build and run pipeline
pipeline = Pipeline(
    name="Lab-ML-Pipeline",
    parameters=[input_data],
    steps=[preprocess_step, train_step, register_step],
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
execution.wait()
print(f"Pipeline execution: {execution.describe()['PipelineExecutionStatus']}")
```

### Verification
- [ ] Pipeline created in SageMaker console
- [ ] All 3 steps complete successfully
- [ ] Model registered in Model Registry
- [ ] Approval status = PendingManualApproval

---

## Lab 2: Distributed Training with DDP

### Objective
Train a model on 4 GPUs across 2 nodes using PyTorch DDP.

### Training Script (train_ddp.py)

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def main():
    # Initialize
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    
    print(f"Rank {rank}/{world_size}, Local Rank {local_rank}")
    
    # Simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).cuda(local_rank)
    
    model = DDP(model, device_ids=[local_rank])
    
    # Synthetic data
    data = torch.randn(10000, 100)
    targets = torch.randint(0, 10, (10000,))
    dataset = TensorDataset(data, targets)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(5):
        sampler.set_epoch(epoch)
        total_loss = 0
        for batch_data, batch_target in loader:
            batch_data = batch_data.cuda(local_rank)
            batch_target = batch_target.cuda(local_rank)
            
            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if rank == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
    
    # Save
    if rank == 0:
        model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        torch.save(model.module.state_dict(), f"{model_dir}/model.pt")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Launch Distributed Training

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train_ddp.py",
    source_dir="./scripts",
    role=role,
    instance_count=2,              # 2 nodes
    instance_type="ml.g5.12xlarge", # 4 GPUs per node
    framework_version="2.1",
    py_version="py310",
    distribution={
        "torch_distributed": {"enabled": True}
    },
)

estimator.fit()
```

### Verification
- [ ] Training runs on 2 nodes (8 GPUs total)
- [ ] NCCL initializes successfully
- [ ] Loss decreases across epochs
- [ ] Model saved by rank 0 only

---

## Lab 3: HyperPod Cluster Setup

### Objective
Create a HyperPod cluster and submit a Slurm training job.

### Steps

```bash
# 1. Create lifecycle scripts
cat > setup_controller.sh << 'EOF'
#!/bin/bash
sudo apt-get update
sudo apt-get install -y slurm-wlm munge
# Configure Slurm controller
sudo systemctl start slurmctld
EOF

cat > setup_worker.sh << 'EOF'
#!/bin/bash
sudo apt-get update
sudo apt-get install -y slurm-wlm nvidia-driver-535 cuda-toolkit-12-2
sudo systemctl start slurmd
# Mount shared filesystem
sudo mount -t lustre $FSX_DNS:/fsx /shared
EOF

# Upload to S3
aws s3 cp setup_controller.sh s3://my-bucket/lifecycle/
aws s3 cp setup_worker.sh s3://my-bucket/lifecycle/

# 2. Create cluster
aws sagemaker create-cluster \
  --cluster-name "lab-hyperpod" \
  --instance-groups '[
    {"InstanceGroupName":"controller","InstanceType":"ml.m5.xlarge","InstanceCount":1,"LifeCycleConfig":{"SourceS3Uri":"s3://my-bucket/lifecycle/","OnCreate":"setup_controller.sh"},"ExecutionRole":"arn:aws:iam::123456:role/HyperPodRole"},
    {"InstanceGroupName":"workers","InstanceType":"ml.g5.12xlarge","InstanceCount":2,"LifeCycleConfig":{"SourceS3Uri":"s3://my-bucket/lifecycle/","OnCreate":"setup_worker.sh"},"ExecutionRole":"arn:aws:iam::123456:role/HyperPodRole"}
  ]'

# 3. Connect to controller
aws ssm start-session --target sagemaker-cluster:lab-hyperpod_controller-i-xxx

# 4. Submit Slurm job
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=test-training
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --output=/shared/logs/%j.out

srun torchrun --nnodes=2 --nproc_per_node=4 /shared/code/train_ddp.py
EOF

# 5. Monitor
squeue
sacct -j <job_id>
```

### Verification
- [ ] Cluster status: InService
- [ ] Can SSH via SSM
- [ ] Slurm job submits and runs
- [ ] GPUs visible from worker nodes

---

## Lab 4: Multi-Model Endpoint

### Objective
Deploy 5 models on a single endpoint, invoke specific models dynamically.

```python
import sagemaker
from sagemaker.multidatamodel import MultiDataModel
import json

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()

# 1. Train 5 simple models and upload to S3
model_prefix = f"s3://{bucket}/mme-lab/models/"

# (Assume you have 5 model.tar.gz files uploaded)
# s3://bucket/mme-lab/models/model-1.tar.gz
# s3://bucket/mme-lab/models/model-2.tar.gz
# ... etc.

# 2. Create Multi-Model Endpoint
multi_model = MultiDataModel(
    name="lab-multi-model",
    model_data_prefix=model_prefix,
    image_uri=sagemaker.image_uris.retrieve("pytorch", session.boto_region_name, "2.1", "inference"),
    role=role,
)

predictor = multi_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="lab-mme-endpoint",
)

# 3. Invoke different models
# Model 1
response_1 = predictor.predict(
    data=json.dumps({"inputs": [1.0, 2.0, 3.0]}),
    target_model="model-1.tar.gz",
)
print(f"Model 1 response: {response_1}")

# Model 3
response_3 = predictor.predict(
    data=json.dumps({"inputs": [4.0, 5.0, 6.0]}),
    target_model="model-3.tar.gz",
)
print(f"Model 3 response: {response_3}")

# 4. List loaded models
print(multi_model.list_models())
```

### Verification
- [ ] Single endpoint deployed
- [ ] Can invoke 5 different models via target_model header
- [ ] Cold start on first invocation (~5-10s)
- [ ] Hot invocation < 50ms

---

## Lab 5: Inference Recommender

### Objective
Find the optimal instance type for your model.

```python
import boto3

sm = boto3.client("sagemaker")

# 1. Create Model Package Group (if not exists)
sm.create_model_package_group(
    ModelPackageGroupName="rec-lab-model",
    ModelPackageGroupDescription="Lab model for inference recommender",
)

# 2. Register model
model_package = sm.create_model_package(
    ModelPackageGroupName="rec-lab-model",
    InferenceSpecification={
        "Containers": [{
            "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1-gpu-py310",
            "ModelDataUrl": "s3://my-bucket/models/my-model/model.tar.gz",
        }],
        "SupportedRealtimeInferenceInstanceTypes": [
            "ml.g5.xlarge", "ml.g5.2xlarge", "ml.g5.4xlarge",
            "ml.p4d.24xlarge", "ml.inf2.xlarge",
        ],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    ModelApprovalStatus="Approved",
)

# 3. Run Inference Recommender
job = sm.create_inference_recommendations_job(
    JobName="lab-rec-job",
    JobType="Default",  # or "Advanced"
    RoleArn=role,
    InputConfig={
        "ModelPackageVersionArn": model_package["ModelPackageArn"],
        "ContainerConfig": {
            "Domain": "MACHINE_LEARNING",
            "Task": "OTHER",
            "Framework": "PYTORCH",
            "FrameworkVersion": "2.1",
            "PayloadConfig": {
                "SamplePayloadUrl": "s3://my-bucket/payload/sample.json",
                "SupportedContentTypes": ["application/json"],
            },
        },
    },
)

# 4. Wait and get results
import time
while True:
    status = sm.describe_inference_recommendations_job(JobName="lab-rec-job")
    if status["Status"] in ["COMPLETED", "FAILED"]:
        break
    time.sleep(60)

# 5. Print recommendations
for rec in status["InferenceRecommendations"]:
    endpoint_config = rec["EndpointConfiguration"]
    metrics = rec["Metrics"]
    print(f"""
    Instance: {endpoint_config['InstanceType']}
    Latency: {metrics['InferenceLatency']}ms
    Cost/hr: ${metrics['CostPerHour']}
    Throughput: {metrics['MaxInvocations']}/min
    """)
```

### Verification
- [ ] Job completes successfully
- [ ] Get recommendations for multiple instance types
- [ ] Can identify best price/performance option

---

## Lab 6: Model Registry with Approval Workflow

### Objective
Register models, set up approval workflow, deploy approved model.

```python
import boto3

sm = boto3.client("sagemaker")

# 1. Create model package group
sm.create_model_package_group(
    ModelPackageGroupName="production-fraud-model",
    ModelPackageGroupDescription="Fraud detection models",
)

# 2. Register v1 (auto-approve for lab)
v1 = sm.create_model_package(
    ModelPackageGroupName="production-fraud-model",
    ModelPackageDescription="Version 1: Baseline model",
    InferenceSpecification={
        "Containers": [{
            "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1-gpu-py310",
            "ModelDataUrl": "s3://my-bucket/models/fraud-v1/model.tar.gz",
        }],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.g5.xlarge"],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    ModelApprovalStatus="Approved",
    CustomerMetadataProperties={
        "accuracy": "0.94",
        "training_date": "2024-01-15",
    },
)

# 3. Register v2 (pending approval)
v2 = sm.create_model_package(
    ModelPackageGroupName="production-fraud-model",
    ModelPackageDescription="Version 2: Improved features",
    InferenceSpecification={
        "Containers": [{
            "Image": "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1-gpu-py310",
            "ModelDataUrl": "s3://my-bucket/models/fraud-v2/model.tar.gz",
        }],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.g5.xlarge"],
        "SupportedContentTypes": ["application/json"],
        "SupportedResponseMIMETypes": ["application/json"],
    },
    ModelApprovalStatus="PendingManualApproval",
    CustomerMetadataProperties={
        "accuracy": "0.96",
        "training_date": "2024-02-01",
    },
)

# 4. List versions
packages = sm.list_model_packages(
    ModelPackageGroupName="production-fraud-model",
    SortBy="CreationTime",
    SortOrder="Descending",
)
for pkg in packages["ModelPackageSummaryList"]:
    print(f"  {pkg['ModelPackageArn']} - {pkg['ModelApprovalStatus']}")

# 5. Approve v2
sm.update_model_package(
    ModelPackageArn=v2["ModelPackageArn"],
    ModelApprovalStatus="Approved",
    ApprovalDescription="A/B test passed. 2% accuracy improvement.",
)

# 6. Deploy approved model
from sagemaker import ModelPackage

model = ModelPackage(
    role=role,
    model_package_arn=v2["ModelPackageArn"],
    sagemaker_session=sagemaker.Session(),
)
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name="fraud-prod-endpoint",
)
```

### Verification
- [ ] Two versions in model registry
- [ ] v1: Approved, v2: Approved (after manual step)
- [ ] Deployed model serves predictions
- [ ] Can trace lineage from endpoint → model → training job

---

## Lab 7: Training Compiler (torch.compile)

### Objective
Compare training speed with and without torch.compile.

```python
# benchmark_compiler.py
import time
import torch
import torch.nn as nn

# Large-ish model for benchmark
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=6,
        )
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))

device = torch.device("cuda")

# Benchmark WITHOUT compile
model = TransformerModel().to(device)
data = torch.randn(64, 128, 512).to(device)

# Warmup
for _ in range(10):
    _ = model(data)
torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    output = model(data)
    loss = output.sum()
    loss.backward()
torch.cuda.synchronize()
time_no_compile = time.time() - start
print(f"Without compile: {time_no_compile:.2f}s for 100 steps")

# Benchmark WITH compile
compiled_model = torch.compile(model, mode="max-autotune")

# Warmup (includes compilation)
for _ in range(10):
    output = compiled_model(data)
    loss = output.sum()
    loss.backward()
torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    output = compiled_model(data)
    loss = output.sum()
    loss.backward()
torch.cuda.synchronize()
time_compiled = time.time() - start
print(f"With compile: {time_compiled:.2f}s for 100 steps")
print(f"Speedup: {time_no_compile/time_compiled:.2f}x")
```

### Verification
- [ ] Both versions run without errors
- [ ] Compiled version is 1.2-1.5x faster
- [ ] Note: First few steps of compiled version are slow (compilation)

---

## Cleanup

```python
# Delete all resources after labs
import boto3
sm = boto3.client("sagemaker")

# Delete endpoints
sm.delete_endpoint(EndpointName="lab-mme-endpoint")
sm.delete_endpoint(EndpointName="fraud-prod-endpoint")

# Delete endpoint configs
sm.delete_endpoint_config(EndpointConfigName="lab-mme-endpoint")

# Delete models
sm.delete_model(ModelName="lab-multi-model")

# Delete cluster (HyperPod)
sm.delete_cluster(ClusterName="lab-hyperpod")

# Delete pipeline
sm.delete_pipeline(PipelineName="Lab-ML-Pipeline")

print("All resources cleaned up!")
```

## Summary Checklist

- [ ] Lab 1: Pipeline (preprocess → train → register) works
- [ ] Lab 2: DDP training on multiple GPUs
- [ ] Lab 3: HyperPod cluster + Slurm job
- [ ] Lab 4: Multi-Model Endpoint with 5 models
- [ ] Lab 5: Inference Recommender finds best instance
- [ ] Lab 6: Model Registry with approval workflow
- [ ] Lab 7: torch.compile speedup measured
