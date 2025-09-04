"""
Integration Example: Prometheus Exporter with BlazeMetrics

Demonstrates how to export BlazeMetrics evaluation metrics to Prometheus for production monitoring.
"""

from blazemetrics.client import BlazeMetricsClient
from blazemetrics.exporters import MetricsExporters

candidates = ["The quick brown fox jumps over the lazy dog."]
references = [["A quick brown fox jumps over the lazy dog."]]

client = BlazeMetricsClient(prometheus_gateway="http://localhost:9091")
exporters = MetricsExporters(prometheus_gateway="http://localhost:9091")

metrics = client.compute_metrics(
    candidates=candidates,
    references=references,
    include=["rouge1", "bleu", "meteor"]
)
exporters.export(metrics, labels={"example": "prometheus"})

print("Metrics exported to Prometheus (if Prometheus Pushgateway is running at localhost:9091).")