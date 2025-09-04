"""
Integration Example: Weights & Biases (wandb) with BlazeMetrics

Demonstrates how to log BlazeMetrics evaluation metrics to wandb for experiment tracking.
"""

import os
import wandb
from blazemetrics.client import BlazeMetricsClient
from blazemetrics.integrations.wandb_integration import BlazeWandbCallback

# Check for WANDB_API_KEY in environment
api_key = os.environ.get("WANDB_API_KEY")
if not api_key:
    print("WANDB_API_KEY not set. Skipping wandb integration test.")
    exit(0)
wandb.login(key=api_key)
wandb.init(project="blazemetrics-demo", name="wandb_integration_example")

candidates = ["The quick brown fox jumps over the lazy dog."]
references = [["A quick brown fox jumps over the lazy dog."]]

client = BlazeMetricsClient()
callback = BlazeWandbCallback(project="blazemetrics-demo", run_name="wandb_integration_example", client=client)

# Compute and log metrics
metrics = client.compute_metrics(
    candidates=candidates,
    references=references,
    include=["rouge1", "bleu", "meteor"]
)
callback.log_text_metrics(candidates, references, step=1, include=["rouge1", "bleu", "meteor"])

print("Metrics logged to wandb:", metrics)
wandb.finish()