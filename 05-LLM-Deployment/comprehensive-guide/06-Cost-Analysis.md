# 06 — Cost Analysis at Different Traffic Levels

## Cost Comparison Table

```
Traffic Level    │ Bedrock Haiku  │ Bedrock Sonnet │ Self-hosted 7B │ Self-hosted 70B
(requests/day)   │ (cheap model)  │ (best quality) │ (g5.xlarge)    │ (p4d.24xlarge)
─────────────────┼────────────────┼────────────────┼────────────────┼────────────────
1K req/day       │ $1/day         │ $14/day        │ $24/day        │ $786/day
10K req/day      │ $11/day        │ $135/day       │ $24/day        │ $786/day
50K req/day      │ $56/day        │ $675/day       │ $24/day        │ $786/day
100K req/day     │ $113/day       │ $1,350/day     │ $48/day (×2)   │ $786/day
500K req/day     │ $563/day       │ $6,750/day     │ $168/day (×7)  │ $786/day
1M req/day       │ $1,125/day     │ $13,500/day    │ $336/day (×14) │ $1,572/day (×2)

Assumptions: 500 input + 200 output tokens per request average
Self-hosted: Fixed cost (always on), scales by adding instances
Bedrock: Linear cost (pay per token)
```

## Break-Even Points

```
Self-hosted 7B (g5.xlarge $1/hr) vs Bedrock Haiku:
├── Self-hosted fixed: $720/month
├── Bedrock break-even: $720 / $0.0011 per request = 654K requests/month
├── = ~22K requests/day
└── Above 22K req/day → self-host is cheaper

Self-hosted 70B (p4d.24xlarge $32.77/hr) vs Bedrock Sonnet:
├── Self-hosted fixed: $23,594/month
├── Bedrock break-even: $23,594 / $0.0135 per request = 1.75M requests/month
├── = ~58K requests/day
└── Above 58K req/day → self-host is cheaper
```

## Cost Optimization Strategies

```
1. Model routing (save 50-70%):
   Simple queries → Haiku/self-hosted 7B (cheap)
   Complex queries → Sonnet (quality)

2. Caching (save 20-40%):
   Same question asked before? Return cached answer.
   Semantic cache: Similar questions → similar answers.

3. Prompt optimization (save 10-30%):
   Shorter system prompts = fewer input tokens.
   Concise instructions = fewer output tokens.

4. Batch inference (save 50%):
   Non-real-time tasks → Bedrock batch (half price).

5. Reserved capacity:
   Predictable traffic → Provisioned Throughput (30-50% off).
```

## Interview Questions

1. **Q: Budget $10K/month. Maximum users kaise serve karoge?**
   A: Hybrid: Self-host LLaMA-8B on 2× g5.xlarge ($1,440/month) for 80% simple queries. Bedrock Haiku ($8,500 remaining budget) for complex 20%. Total capacity: ~200K+ requests/day.

2. **Q: Cost spike detect kaise karoge?**
   A: CloudWatch billing alarm + per-user token tracking + daily budget caps. Alert at 80% of monthly budget.
