"""
Example: MLflow Exporter with BlazeMetrics

This script demonstrates how to use BlazeMetrics with MLflow for experiment tracking.
"""

from blazemetrics.client import BlazeMetricsClient
from blazemetrics.integrations.mlflow_integration import BlazeMLflowRun

import mlflow

candidates = ["The quick brown fox jumps over the lazy dog."]
references = [["A quick brown fox jumps over the lazy dog."]]

client = BlazeMetricsClient()

with BlazeMLflowRun(run_name="blazemetrics_example") as run:
    metrics = client.compute_metrics(
        candidates=candidates,
        references=references,
        include=["rouge1", "bleu", "meteor"]
    )
    print("Metrics:", metrics)
    mlflow.log_metrics(metrics)