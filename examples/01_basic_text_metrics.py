"""
01_basic_text_metrics.py

BlazeMetrics Example – Basic Text Evaluation
-------------------------------------------
This example demonstrates the two main ways to compute text metrics in BlazeMetrics:
    1. **Default (all metrics):** Just call `compute_metrics` -- returns all key metrics (ROUGE, BLEU, CHRF, METEOR, WER, etc)
    2. **Custom set:** You can specify `include=[...]` to select just the metrics you want, for speed/minimalism.

Supported metric names are: 'rouge1', 'rouge2', 'rougeL', 'bleu', 'chrf', 'meteor', 'wer', 'token_f1', 'jaccard'

Expected output:
  - Prints each metric per sample and overall aggregate values.
  - No guardrails or analytics—pure evaluation only.
"""
from blazemetrics import BlazeMetricsClient

# Example predictions (candidates) and their references (ground truths)
candidates = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this is a test."
]
references = [
    ["The fast brown fox jumps over the lazy dog."],   # ref for first candidate
    ["Hello World! This is only a test."]             # ref for second
]

client = BlazeMetricsClient()

# === EXAMPLE 1: DEFAULT METRICS LIST ===
# If you do not specify any metrics, BlazeMetrics will compute all text-generation metrics:
# ["rouge1", "rouge2", "rougeL", "bleu", "chrf", "meteor", "wer", "token_f1", "jaccard"]
metrics_default = client.compute_metrics(candidates, references)
agg_default = client.aggregate_metrics(metrics_default)

print("\n==== Sample-wise metrics (ALL DEFAULTS) ====")
for k, vals in metrics_default.items():
    print(f"  {k}: {vals}")
print("\nAggregate metrics (ALL DEFAULTS):")
for k, v in agg_default.items():
    print(f"  {k}: {v:.3f}")

# === EXAMPLE 2: ONLY SELECTED METRICS ===
# If you only care about a few metrics (for example, BLEU and WER), specify them as follows:
metrics_selected = client.compute_metrics(
    candidates, references, include=["bleu", "wer"]
)
agg_selected = client.aggregate_metrics(metrics_selected)

print("\n==== Sample-wise metrics (BLEU, WER only) ====")
for k, vals in metrics_selected.items():
    print(f"  {k}: {vals}")
print("\nAggregate metrics (BLEU, WER only):")
for k, v in agg_selected.items():
    print(f"  {k}: {v:.3f}")

# ---
# Detailed explanation:
# - By default, BlazeMetrics computes all supported metrics. Use this for completeness and benchmarks.
# - For large-scale or minimal evaluation (e.g., just BLEU for translation), pass the `include` argument
#   with a list of names from this set: ["rouge1", "rouge2", "rougeL", "bleu", "chrf", "meteor", "wer", "token_f1", "jaccard"]
# - The results dictionary will only include those metrics requested.
# This makes BlazeMetrics fast, flexible, and trustable for every client use-case.
