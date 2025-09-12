"""
02_metrics_exporters.py

BlazeMetrics Example – Real-World Metrics Export to Prometheus/StatsD
---------------------------------------------------------------------
This practical, ready-to-run example shows how to push *actual BlazeMetrics text evaluation metrics* to monitoring.
Prometheus can scrape the Pushgateway for metrics visualization and alerting (Grafana, Datadog, etc).

How it works:
  1. Run BlazeMetrics evaluation on your models/outputs (candidates/references).
  2. Push the real metric scores (BLEU, ROUGE, WER, etc.) to Prometheus/StatsD with contextual labels for model/version/env/job.

**About Pushgateway:**
- BlazeMetrics uses the [Prometheus Pushgateway](https://github.com/prometheus/pushgateway) for exporting model batch/job metrics. Pushgateway is designed for metrics from ephemeral jobs (like model batch evaluations), not for continuously running apps: Prometheus then scrapes and stores these for dashboards and alerting.

Setup notes:
  - Prometheus Pushgateway must be running and reachable (localhost:9091 in this example).
  - StatsD server should be listening on the given port, or comment that line out.
  - Customize label values to match your ML models/CI infra for production.

"""
# EXAMPLE 1: Export ALL main metrics (default)
from blazemetrics import BlazeMetricsClient, MetricsExporters

candidates = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this is a test."
]
references = [
    ["The fast brown fox jumps over the lazy dog."],
    ["Hello World! This is only a test."]
]

client = BlazeMetricsClient()
metrics = client.compute_metrics(candidates, references)
agg_metrics = client.aggregate_metrics(metrics)

labels = {
    "model": "llama-3-prod",
    "job": "llm_eval_batch_42",
    "env": "production",
    "owner": "blazemetrics"
}

exporter = MetricsExporters(
    prometheus_gateway="localhost:9091",   # Local Pushgateway for demo/testing
    statsd_addr="localhost:8125"           # Comment out if no StatsD running
)
exporter.export(agg_metrics, labels=labels)

print("Exported ALL BlazeMetrics metrics to Prometheus (Pushgateway)/StatsD:")
for k, v in agg_metrics.items():
    print(f"  {k}: {v:.3f}")
print("Check http://localhost:9091/metrics for your pushed metrics!")
print("If using Prometheus + Grafana, add 'blazemetrics_*' to your panels/queries.")

# ---
# EXAMPLE 2: Export ONLY selected metrics (e.g. BLEU and WER only)
# You can restrict which metrics are computed/exported by passing the 'include' argument to compute_metrics().
# This is useful for lightweight monitoring or runtime-critical endpoints.
metrics_selected = client.compute_metrics(
    candidates, references, include=["bleu", "wer"]
)
agg_selected = client.aggregate_metrics(metrics_selected)

labels_selected = dict(labels)
labels_selected["job"] = "llm_eval_bleu_wer_only"
exporter.export(agg_selected, labels=labels_selected)

print("\nExported ONLY BLEU/WER metrics to Prometheus (Pushgateway)/StatsD:")
for k, v in agg_selected.items():
    print(f"  {k}: {v:.3f}")
print("Check http://localhost:9091/metrics for your specific metric subset!")

# -------- USAGE IN REAL WORLD PIPELINES -------------
# - Call this export at the end of each evaluation/validation/test batch.
# - You can export ALL available metrics, or a minimal set (e.g., just BLEU, WER, etc.).
# - Use detailed labels for slicing metrics by model/version/env/run in dashboards/alerts.
# - Prometheus scrapes the Pushgateway, collects these time-series for downstream use.
# - All keys become Prometheus/StatsD metrics prefixed with 'blazemetrics_'.
# ----------------------------------------------------
