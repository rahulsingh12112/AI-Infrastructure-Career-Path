# 03 — Bedrock vs Self-Hosted Decision Framework

## Quick Decision Matrix

```
Factor              │ Bedrock Wins When           │ Self-Hosted Wins When
────────────────────┼─────────────────────────────┼──────────────────────────
Cost (low traffic)  │ < 1M tokens/day             │ > 5M tokens/day
Cost (high traffic) │ Unpredictable bursts        │ Steady high volume
Time to market      │ Need it TODAY               │ Can wait 1-2 weeks
Model choice        │ Want Claude/multiple models │ Open-weight model is enough
Customization       │ Standard use cases          │ Custom serving logic needed
Latency             │ ~200-500ms acceptable       │ Need < 100ms
Data privacy        │ AWS region is sufficient    │ Need on-prem/air-gapped
Team expertise      │ No ML infra team            │ Have GPU/K8s expertise
Scaling             │ Auto (AWS handles)          │ Manual (you manage)
Fine-tuning         │ Bedrock fine-tuning enough  │ Need full RLHF/custom training
```

## Detailed Cost Comparison

```
Scenario: Customer support chatbot
├── 100K requests/day
├── Average: 500 input tokens + 200 output tokens per request
├── Total: 50M input + 20M output tokens/day

BEDROCK (Claude 3.5 Sonnet):
├── Input: 50M × $3.00/1M = $150/day
├── Output: 20M × $15.00/1M = $300/day
├── Total: $450/day = $13,500/month

BEDROCK (Claude 3 Haiku — if quality sufficient):
├── Input: 50M × $0.25/1M = $12.50/day
├── Output: 20M × $1.25/1M = $25/day
├── Total: $37.50/day = $1,125/month

SELF-HOSTED (LLaMA-70B on p4d.24xlarge):
├── Instance: $32.77/hr × 24 × 30 = $23,594/month
├── But can handle much MORE traffic (up to 2500 tok/s)
├── At 2500 tok/s = 216M tokens/day capacity
├── Cost per same 70M tokens: effectively $7,600/month
├── Savings vs Sonnet: 44%
├── But: Need infra team, monitoring, on-call

SELF-HOSTED (LLaMA-70B AWQ on g5.48xlarge):
├── Instance: $16.29/hr × 24 × 30 = $11,729/month
├── Handles ~800 tok/s = sufficient for this traffic
├── Cost for same workload: ~$11,729/month
├── Savings vs Sonnet: 13% (not much!)
├── Better with Haiku comparison: 10x MORE expensive!

CONCLUSION for this scenario:
├── Low traffic → Bedrock Haiku ($1,125/month) — clear winner
├── Need Claude quality → Bedrock Sonnet ($13,500/month)
├── Want full control + scale → Self-hosted ($7,600-11,729/month)
└── At 10x traffic → Self-hosted becomes much cheaper
```

## Break-Even Analysis

```
At what traffic level does self-hosting become cheaper?

Self-hosted LLaMA-70B (p4d.24xlarge):
├── Fixed cost: ~$24,000/month (always on)
├── Capacity: ~2500 tokens/sec = 6.5B tokens/month
├── Cost per 1M tokens: $24,000 / 6,500 = $3.69/1M

Bedrock Claude 3.5 Sonnet:
├── Variable cost: $3 input + $15 output per 1M
├── Blended (assuming 70% input, 30% output): ~$6.60/1M

Break-even: When monthly Bedrock cost > $24,000
= $24,000 / $6.60 per 1M = 3.6B tokens/month
= ~120M tokens/day
= ~170K requests/day (700 tokens avg)

Rule of thumb:
├── < 50K requests/day → Bedrock (cheaper, no ops)
├── 50K-200K requests/day → Depends on model quality needs
├── > 200K requests/day → Self-hosted (significantly cheaper)
├── Unpredictable traffic → Bedrock (pay per use)
└── GPU team available → Self-hosted threshold lower
```

## Decision Framework (Flowchart)

```
START
│
├── Need Claude/GPT-4 class quality?
│   ├── YES → Bedrock (Claude) or OpenAI
│   │   └── Traffic > 200K req/day? → Consider Provisioned Throughput
│   └── NO → Open-weight model (LLaMA/Mistral) sufficient?
│       ├── YES → Self-host consideration ↓
│       └── NO → Bedrock with appropriate model
│
├── Have ML Infra team (K8s + GPU expertise)?
│   ├── YES → Self-hosting viable
│   └── NO → Bedrock (avoid ops complexity)
│
├── Traffic level?
│   ├── < 50K req/day → Bedrock (always cheaper)
│   ├── 50K-200K req/day → Cost analysis needed
│   └── > 200K req/day → Self-host (much cheaper)
│
├── Latency requirements?
│   ├── < 100ms TTFT → Self-host (no API gateway hop)
│   └── 200-500ms OK → Either works
│
├── Data sensitivity?
│   ├── Can't leave VPC at all → Self-host
│   ├── AWS region OK → Bedrock (with VPC endpoint)
│   └── Need air-gapped → Self-host on-prem
│
├── Need multiple models?
│   ├── YES (A/B test Claude vs LLaMA) → Bedrock
│   └── NO (one model is fine) → Either
│
└── DECISION
```

## Hybrid Architecture (Best of Both)

```
Most production systems use BOTH:

┌─────────────────────────────────────────────────────────┐
│              Hybrid LLM Architecture                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Request Router                                          │
│  ├── Simple queries → Self-hosted LLaMA-8B (cheap)      │
│  ├── Complex queries → Bedrock Claude Sonnet (quality)  │
│  ├── Coding tasks → Self-hosted CodeLlama (specialized)│
│  └── Overflow → Bedrock (burst capacity)                │
│                                                          │
│  Benefits:                                               │
│  ├── 70% of traffic on cheap self-hosted (cost)         │
│  ├── 30% complex queries get best quality (Claude)      │
│  ├── Burst traffic handled by Bedrock (no capacity issue)│
│  └── Total cost: 50% less than all-Bedrock              │
└─────────────────────────────────────────────────────────┘
```

## Migration Path (Bedrock → Self-hosted)

```
Phase 1: Start with Bedrock (Week 1)
├── Validate use case, measure token usage
├── Understand traffic patterns
└── No infra investment

Phase 2: Parallel testing (Week 2-4)
├── Deploy self-hosted LLaMA on EKS (dev)
├── Shadow traffic (send to both, compare)
├── Evaluate quality difference
└── Measure cost at actual traffic

Phase 3: Gradual migration (Week 4-8)
├── Route 10% traffic to self-hosted
├── Monitor quality + latency
├── Increase to 50%, then 100%
├── Keep Bedrock as fallback
└── Switch routing if self-hosted issues

Phase 4: Optimize (Month 2+)
├── Fine-tune model for your domain
├── Quantize for better throughput
├── Optimize serving (prefix caching, batching)
└── Reduce Bedrock to burst-only
```

## Interview Questions

1. **Q: New startup, 10K daily active users, chatbot bana rahe hain. Bedrock ya self-hosted?**
   A: Bedrock. Reasons: (1) 10K users ≈ 30-50K requests/day = well within Bedrock's sweet spot, (2) Startup = small team, can't afford GPU ops engineer, (3) Need to iterate fast (try different models), (4) Predictable per-token cost (no wasted GPU during low traffic hours), (5) Start with Haiku ($1K/month), upgrade to Sonnet if quality needs increase.

2. **Q: Enterprise, 500K requests/day, already have K8s team. Recommendation?**
   A: Hybrid. (1) Self-host LLaMA-70B for 80% of traffic (bulk queries = $15K/month), (2) Bedrock Claude for complex 20% (reasoning, analysis = $3K/month), (3) Total: $18K vs all-Bedrock $90K+. (4) Use Bedrock as overflow/fallback. (5) Save $70K+/month = pays for infra engineer salary.

3. **Q: Self-hosted model ki quality Bedrock Claude se kam hai. Kya karein?**
   A: (1) Fine-tune LLaMA on your domain data (significant quality boost), (2) Use larger model (70B instead of 8B), (3) Better prompting (system prompts optimized for LLaMA), (4) RAG pipeline (ground in your data), (5) If still not enough: Use self-hosted for simple tasks, route complex ones to Bedrock. Don't force self-hosted where quality matters.

4. **Q: Bedrock ka cost suddenly spike hua. Debug kaise karoge?**
   A: (1) CloudWatch metrics: Check token count per API call, (2) Are output tokens too high? (model being verbose = reduce max_tokens), (3) Retry loops? (errors causing re-sends), (4) New feature launched? (more users/longer conversations), (5) Prompt injection? (adversarial users making long outputs). Fix: Token budgets per user, output length limits, monitoring alerts on cost anomalies.
