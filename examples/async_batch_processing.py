"""
Example: Async Batch Processing with BlazeMetrics

Demonstrates how to use BlazeMetrics' async APIs for large-scale, concurrent evaluation.
"""

import asyncio
from blazemetrics.client import BlazeMetricsClient

# Simulate a large batch of candidate/reference pairs
candidates = [f"Sample candidate {i}" for i in range(100)]
references = [[f"Sample reference {i}"] for i in range(100)]

client = BlazeMetricsClient()

async def async_compute_metrics():
    loop = asyncio.get_event_loop()
    # Run compute_metrics in a thread pool for async compatibility
    results = await loop.run_in_executor(None, client.compute_metrics, candidates, references, ["rouge1", "bleu"])
    print("Async Batch Metrics:", results)

if __name__ == "__main__":
    asyncio.run(async_compute_metrics())