# 01 — AWS Bedrock: Models, APIs, Guardrails

## What is Amazon Bedrock?

Fully managed service to access foundation models (FMs) via API. No infrastructure to manage.

```
Bedrock = "LLM as a Service":
├── Pay per token (no GPU management)
├── Multiple model providers (Anthropic, Meta, Amazon, Cohere, etc.)
├── No model hosting, scaling, or maintenance
├── Built-in: Guardrails, RAG (Knowledge Bases), Agents, Fine-tuning
├── Data never used to train models (privacy)
└── Enterprise features: VPC endpoints, CloudTrail, IAM
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Amazon Bedrock                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Your Application                                            │
│       │ (API call)                                           │
│       ▼                                                      │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Bedrock API Gateway                              │       │
│  │  ├── Authentication (IAM/SigV4)                   │       │
│  │  ├── Rate limiting                                │       │
│  │  ├── Guardrails (content filtering)              │       │
│  │  └── Routing to model provider                   │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         │                                    │
│       ┌─────────────────┼─────────────────┐                 │
│       ▼                 ▼                 ▼                  │
│  ┌─────────┐     ┌─────────┐      ┌─────────┐             │
│  │Anthropic│     │  Meta   │      │ Amazon  │             │
│  │Claude 3 │     │LLaMA 3 │      │ Titan   │             │
│  │Sonnet/  │     │         │      │         │             │
│  │Opus/Haiku│    │         │      │         │             │
│  └─────────┘     └─────────┘      └─────────┘             │
│                                                              │
│  Features:                                                   │
│  ├── Knowledge Bases (RAG with vector DB)                   │
│  ├── Agents (multi-step tool use)                           │
│  ├── Guardrails (safety, PII filtering)                     │
│  ├── Fine-tuning (custom model on your data)                │
│  ├── Model evaluation                                        │
│  └── Prompt management                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Available Models

```
Provider    │ Model             │ Context │ Strength          │ Price (input/output per 1M tokens)
────────────┼───────────────────┼─────────┼───────────────────┼──────────────────────────────────
Anthropic   │ Claude 3.5 Sonnet │ 200K    │ Best overall      │ $3.00 / $15.00
Anthropic   │ Claude 3 Haiku    │ 200K    │ Fast, cheap       │ $0.25 / $1.25
Anthropic   │ Claude 3 Opus     │ 200K    │ Most capable      │ $15.00 / $75.00
Meta        │ LLaMA 3.1 70B     │ 128K    │ Open-weight, good │ $2.65 / $3.50
Meta        │ LLaMA 3.1 8B      │ 128K    │ Small, fast       │ $0.22 / $0.22
Amazon      │ Titan Text Premier│ 32K     │ AWS-native        │ $0.50 / $1.50
Amazon      │ Titan Embeddings  │ 8K      │ Embeddings        │ $0.02 / -
Cohere      │ Command R+        │ 128K    │ RAG optimized     │ $3.00 / $15.00
Mistral     │ Mistral Large     │ 128K    │ EU compliance     │ $4.00 / $12.00
AI21        │ Jamba 1.5 Large   │ 256K    │ Long context      │ $2.00 / $8.00
```

## Using Bedrock API

### Basic Invocation
```python
import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Claude 3.5 Sonnet
response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "What is machine learning?"}
        ],
        "temperature": 0.7,
    }),
)

result = json.loads(response["body"].read())
print(result["content"][0]["text"])
```

### Streaming Response
```python
response = bedrock.invoke_model_with_response_stream(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Write a poem about clouds"}],
    }),
)

stream = response.get("body")
for event in stream:
    chunk = event.get("chunk")
    if chunk:
        data = json.loads(chunk.get("bytes").decode())
        if data["type"] == "content_block_delta":
            print(data["delta"]["text"], end="", flush=True)
```

### Converse API (Unified across models)
```python
# Converse API: Same code works for ALL models
response = bedrock.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[
        {"role": "user", "content": [{"text": "What is AI?"}]}
    ],
    inferenceConfig={
        "maxTokens": 512,
        "temperature": 0.7,
        "topP": 0.9,
    },
)

print(response["output"]["message"]["content"][0]["text"])
print(f"Input tokens: {response['usage']['inputTokens']}")
print(f"Output tokens: {response['usage']['outputTokens']}")
```

## Bedrock Guardrails

```python
# Create guardrail
bedrock_client = boto3.client("bedrock", region_name="us-east-1")

guardrail = bedrock_client.create_guardrail(
    name="production-guardrail",
    description="Content safety for production",
    
    # Topic filters (block specific topics)
    topicPolicyConfig={
        "topicsConfig": [
            {
                "name": "Financial Advice",
                "definition": "Specific financial investment recommendations",
                "type": "DENY",
            },
        ],
    },
    
    # Content filters (hate, violence, sexual, etc.)
    contentPolicyConfig={
        "filtersConfig": [
            {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
            {"type": "INSULTS", "inputStrength": "MEDIUM", "outputStrength": "HIGH"},
        ],
    },
    
    # PII handling
    sensitiveInformationPolicyConfig={
        "piiEntitiesConfig": [
            {"type": "EMAIL", "action": "ANONYMIZE"},
            {"type": "PHONE", "action": "ANONYMIZE"},
            {"type": "SSN", "action": "BLOCK"},
        ],
    },
    
    blockedInputMessaging="I cannot process this request.",
    blockedOutputsMessaging="I cannot provide that information.",
)

# Use guardrail with model
response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    guardrailIdentifier=guardrail["guardrailId"],
    guardrailVersion="DRAFT",
    body=json.dumps({...}),
)
```

## Bedrock Knowledge Bases (RAG)

```python
# RAG = Retrieval-Augmented Generation
# Your documents + LLM = Answers grounded in your data

bedrock_agent = boto3.client("bedrock-agent", region_name="us-east-1")

# Create knowledge base
kb = bedrock_agent.create_knowledge_base(
    name="company-docs-kb",
    roleArn="arn:aws:iam::123456:role/BedrockKBRole",
    knowledgeBaseConfiguration={
        "type": "VECTOR",
        "vectorKnowledgeBaseConfiguration": {
            "embeddingModelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0",
        },
    },
    storageConfiguration={
        "type": "OPENSEARCH_SERVERLESS",
        "opensearchServerlessConfiguration": {
            "collectionArn": "arn:aws:aoss:us-east-1:123456:collection/my-collection",
            "vectorIndexName": "docs-index",
            "fieldMapping": {
                "vectorField": "embedding",
                "textField": "text",
                "metadataField": "metadata",
            },
        },
    },
)

# Query knowledge base
bedrock_agent_runtime = boto3.client("bedrock-agent-runtime")

response = bedrock_agent_runtime.retrieve_and_generate(
    input={"text": "What is our refund policy?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": kb["knowledgeBase"]["knowledgeBaseId"],
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
        },
    },
)

print(response["output"]["text"])
# Includes citations from your documents!
```

## Bedrock Fine-tuning

```python
# Fine-tune a model on your data
bedrock_client = boto3.client("bedrock", region_name="us-east-1")

# Training data format (JSONL in S3):
# {"prompt": "Classify: ...", "completion": "positive"}

job = bedrock_client.create_model_customization_job(
    jobName="my-finetuning-job",
    customModelName="my-custom-claude",
    roleArn="arn:aws:iam::123456:role/BedrockFinetuneRole",
    baseModelIdentifier="anthropic.claude-3-haiku-20240307-v1:0",
    
    trainingDataConfig={
        "s3Uri": "s3://my-bucket/training-data/train.jsonl",
    },
    validationDataConfig={
        "validators": [{"s3Uri": "s3://my-bucket/training-data/val.jsonl"}],
    },
    outputDataConfig={
        "s3Uri": "s3://my-bucket/output/",
    },
    
    hyperParameters={
        "epochCount": "3",
        "batchSize": "8",
        "learningRate": "0.00001",
    },
)

# After training: Use custom model via Provisioned Throughput
```

## Provisioned Throughput (Reserved Capacity)

```python
# For predictable high-volume workloads: Reserve model capacity

bedrock_client.create_provisioned_model_throughput(
    modelUnits=1,           # 1 model unit (specific tokens/min capacity)
    provisionedModelName="my-reserved-claude",
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    commitmentDuration="OneMonth",  # or "SixMonths" for discount
)

# Pricing: Fixed monthly cost regardless of usage
# Good when: Predictable high traffic (cheaper than on-demand at scale)
# Bad when: Variable/low traffic (paying for unused capacity)
```

## Bedrock Pricing Model

```
On-Demand (pay per token):
├── No upfront commitment
├── Best for: Variable traffic, experimentation
├── Claude 3.5 Sonnet: $3/1M input + $15/1M output
└── LLaMA 3.1 8B: $0.22/1M input + $0.22/1M output

Provisioned Throughput (reserved):
├── Fixed monthly cost for guaranteed capacity
├── Best for: Predictable high-volume production
├── 1-6 month commitments (deeper discounts)
└── ~30-50% cheaper than on-demand at high volume

Batch Inference:
├── 50% discount vs on-demand
├── Best for: Non-real-time processing (overnight jobs)
├── Submit batch, get results in hours
└── Great for: Content generation, data processing
```

## Interview Questions

1. **Q: Bedrock vs self-hosted LLM — quick decision criteria?**
   A: Bedrock when: (1) Don't want to manage GPUs, (2) Need multiple models (A/B test Claude vs LLaMA), (3) Variable/unpredictable traffic, (4) Fast time-to-market, (5) Compliance needs (data stays in AWS, no model training on your data). Self-hosted when: (1) Need maximum customization, (2) Very high volume (cheaper at scale), (3) Specific model requirements (custom fine-tunes), (4) Latency-sensitive (no API gateway overhead).

2. **Q: Bedrock pe Claude use kar rahe ho, cost $50K/month hai. Kaise reduce?**
   A: (1) Switch long-context queries to Haiku (10x cheaper, good for simple tasks), (2) Implement prompt caching (Bedrock supports it for repeated prefixes), (3) Provisioned Throughput if traffic predictable (30-50% savings), (4) Batch inference for non-real-time workloads (50% off), (5) Reduce output tokens (stricter max_tokens), (6) Consider self-hosting LLaMA for simpler tasks.

3. **Q: Guardrails production mein kaise implement karoge?**
   A: (1) Create guardrail with content filters (HATE, VIOLENCE, SEXUAL = HIGH), (2) Add PII detection (ANONYMIZE emails/phones, BLOCK SSN), (3) Add topic denial (specific blocked topics for your domain), (4) Apply guardrail to all invoke_model calls, (5) Monitor guardrail interventions in CloudWatch, (6) Test with adversarial inputs before launch.

4. **Q: Bedrock Knowledge Base vs custom RAG (Pinecone + LangChain)?**
   A: Bedrock KB: Managed, less code, automatic chunking/embedding/retrieval, integrates with S3 data sources, limited customization. Custom RAG: Full control over chunking strategy, embedding model, retrieval algorithm, re-ranking. Choose Bedrock KB for quick setup + standard use cases. Choose custom RAG when you need fine-grained control or specialized retrieval logic.
