"""
10_complete_llm_workflow.py

BlazeMetrics Example – Production-Quality End-to-End LLM Evaluation Pipeline
----------------------------------------------------------------------------
Covers every major BlazeMetrics feature for *enterprise* QA:

  - Classical text metrics (BLEU, ROUGE, WER, etc.); customizable metrics/normalization
  - Guardrails check (compliance, PII, regex/policy blocklist, unsafe pattern scan)
  - Real-time analytics window/summary (trending, alerting, anomaly detection)
  - Factuality/hallucination scoring (LLM-based, eg OpenAI judge; not dummy rules)
  - Rich Model Card reporting, including config, violations, and provenance trace

Parameters, API and best practices are all doc'ed inline.
"""

from blazemetrics import BlazeMetricsClient
from blazemetrics.llm_judge import LLMJudge
import os

# 1. Define your batch of LLM generations and references.
candidates = [
    "Alice's email is alice@example.com.",
    "Paris is the capital city of France.",
    "2 + 2 is 5.",
    "You should buy Bitcoin on my advice.",
]
references = [
    ["Her email is alice@example.com."],
    ["Paris is the capital of France."],
    ["2 + 2 = 4"],
    [""]
]

# 2. Configure BlazeMetricsClient.
client = BlazeMetricsClient(
    # --- Guardrails and compliance ---
    blocklist=["Bitcoin"],                   # Block/flag use of "Bitcoin" in output
    redact_pii=True,                        # Redact personally identifiable information
    regexes=[r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"],  # Also block regex matches for email addresses
    case_insensitive=True,
    # --- Analytics/monitoring ---
    enable_analytics=True,
    analytics_window=100,
    analytics_alerts=True,                  # Enable streaming alerts
    analytics_anomalies=True,               # Enable streaming anomaly detection
    analytics_trends=True,                  # Enable per-metric trending summary
    # --- Other optional params ---
    metrics_include=["bleu", "rouge1", "wer", "meteor", "chrf", "token_f1", "jaccard"],
    metrics_lowercase=True,                 # Normalize case for metrics
    metrics_stemming=False,                 # Optionally enable stemming
)

# 3. Metrics evaluation
metrics = client.compute_metrics(
    candidates, references,
    include=client.config.metrics_include,
    lowercase=client.config.metrics_lowercase,
    stemming=client.config.metrics_stemming,
)
agg = client.aggregate_metrics(metrics)

# 4. Safety checks (guardrails, PII, blocklist, regex, etc.)
safety = client.check_safety(candidates)

# 5. Add to analytics window; get analytic summary (trends, perf)
client.add_metrics(agg)
analytics = client.get_analytics_summary()

# 6. Factuality/hallucination scoring with an LLM-based judge!
#    - Uses OpenAI/Anthropic or custom endpoint (set env OPENAI_API_KEY)
judge = LLMJudge(
    provider="openai",                   # Or 'anthropic'
    api_key=os.getenv("OPENAI_API_KEY"), # Should be set in your environment
    model="gpt-4o",               # Or "gpt-4" for higher accuracy
)
# NOTE: This will consume OpenAI API credits and needs your API key set.
def openai_factuality_judge(output, reference):
    # LLMJudge expects batched input, so we wrap and unwrap
    result = judge.score([output], [reference])
    # Return whatever the judge/LLM returns, but ensure a factuality/hallucination key is present
    return {
        **result[0],
        "factuality": result[0].get("faithfulness", 0.0),
        "hallucination": result[0].get("hallucination", 1.0 - result[0].get("faithfulness", 0.0)),
    }

client.set_factuality_scorer(openai_factuality_judge)
facts = client.evaluate_factuality(
    candidates,
    [r[0] for r in references]
)

# 7. Model card with all details, violations, factuality, etc.
model_card = client.generate_model_card(
    "sample-llm-e2e", metrics, analytics, {"config_used": client.config.__dict__},
    violations=safety, factuality=facts, provenance=[]
)

print("\n--- Model Card Example (Markdown) ---")
print(model_card)

"""
-----------------------
Parameter/Feature Explanation
-----------------------

- BlazeMetricsClient(...):
    blocklist         # Phrases to block/flag (eg 'Bitcoin')
    redact_pii        # Redact detected emails, phones, and other PII (code/sys patterns)
    regexes           # Custom regex policy checks, e.g. catch emails, CC #s, etc.
    metrics_include   # Metrics to compute ("bleu", "rouge1", "wer", etc.)
    metrics_lowercase # Normalize case before metrics computing
    metrics_stemming  # Optionally enable word stemming (reduce bias in scoring)
    enable_analytics/analytics_*: # Control BlazeMetrics' real-time performance analytics

- LLMJudge(...):
    provider     # 'openai', 'anthropic', or callable for custom judge/scorer
    model        # OpenAI/Anthropic model (eg gpt-4o, claude-3)
    system_prompt# Optional; default asks for 'faithfulness', 'hallucination', 'bias'
    api_key      # Your LLM vendor API key (must be provided for remote scoring)

- .set_factuality_scorer(fn):
    Flexible: the function can call any LLM or judge, or custom logic.

*Fact Scoring* is now **production-grade**—no dummy logic required!

"""