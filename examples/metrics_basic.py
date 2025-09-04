"""
Example: Basic Text Metrics with BlazeMetrics

This script demonstrates how to compute common NLP metrics (ROUGE, BLEU, METEOR, etc.) using BlazeMetrics.
"""

from blazemetrics.client import BlazeMetricsClient

# Example candidate and reference texts
candidates = ["The quick brown fox jumps over the lazy dog."]
references = [["A quick brown fox jumps over the lazy dog."]]

client = BlazeMetricsClient()

metrics = client.compute_metrics(
    candidates=candidates,
    references=references,
    include=["rouge1", "rouge2", "rougeL", "bleu", "meteor", "wer", "token_f1", "jaccard"]
)

print("Metrics:", metrics)