"""
Example: Distributed Processing with Ray and BlazeMetrics

Demonstrates how to use Ray for distributed, parallel evaluation of large batches with BlazeMetrics.
"""

import ray
from blazemetrics.client import BlazeMetricsClient

ray.init(ignore_reinit_error=True)

@ray.remote
def compute_metrics_batch(candidates, references):
    client = BlazeMetricsClient()
    return client.compute_metrics(candidates, references, include=["rouge1", "bleu"])

# Simulate a large dataset split into chunks
N = 1000
chunk_size = 100
candidates = [f"Sample candidate {i}" for i in range(N)]
references = [[f"Sample reference {i}"] for i in range(N)]

batches = [
    (candidates[i:i+chunk_size], references[i:i+chunk_size])
    for i in range(0, N, chunk_size)
]

futures = [compute_metrics_batch.remote(c, r) for c, r in batches]
results = ray.get(futures)

# Aggregate results
all_rouge1 = [score for batch in results for score in batch["rouge1_f1"]]
all_bleu = [score for batch in results for score in batch["bleu"]]

print("Ray Distributed ROUGE1 (first 5):", all_rouge1[:5])
print("Ray Distributed BLEU (first 5):", all_bleu[:5])