"""
Showcase: End-to-End LLM Evaluation Pipeline with BlazeMetrics

This script demonstrates a full workflow: generation, evaluation, guardrails, streaming analytics, and exporter integration.
"""

from blazemetrics.client import BlazeMetricsClient
from blazemetrics.streaming_analytics import StreamingAnalytics, AlertRule
from blazemetrics.integrations.mlflow_integration import BlazeMLflowRun
from transformers import pipeline

import mlflow

# 1. Generate text with HuggingFace
hf_pipeline = pipeline("text-generation", model="gpt2")
prompt = "The future of AI is"
generated = hf_pipeline(prompt, max_new_tokens=20)[0]["generated_text"]

# 2. Evaluate with BlazeMetrics
candidates = [generated]
references = [["The future of AI is bright and full of possibilities."]]

client = BlazeMetricsClient()
metrics = client.compute_metrics(
    candidates=candidates,
    references=references,
    include=["rouge1", "bleu", "meteor", "wer"]
)
print("Metrics:", metrics)

# 3. Guardrails
safety = client.check_safety(candidates)
print("Safety:", safety)

# 4. Streaming analytics
analytics = StreamingAnalytics(
    window_size=5,
    alert_rules=[AlertRule(metric_name="rouge1", threshold=0.5, severity="high")]
)
analytics.add_metrics(metrics)
print("Streaming Analytics Summary:", analytics.get_metric_summary())

# 5. Export to MLflow
with BlazeMLflowRun(run_name="llm_e2e_showcase") as run:
    mlflow.log_metrics(metrics)
    print("Metrics logged to MLflow.")