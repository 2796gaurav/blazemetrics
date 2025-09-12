"""
03_async_parallel_metrics.py

BlazeMetrics Example – Async & Parallel Metrics Computation (Production-Grade)
------------------------------------------------------------------------------
Real-world use case: Compute *only the specific metrics* required for monitoring, SLA, or reporting,
rather than all available metrics by default. This is the pattern you would use for a
production LLM pipeline, async API, or high-throughput batch job.

Main Concepts:
  - Use `include` to specify subset of metrics (e.g. just BLEU for translation, or only WER for ASR)
  - Use `compute_metrics_async` for asyncio-powered async logic (serving, streaming, UI, etc.)
  - Use `compute_metrics_parallel` for thread-pooled multi-core throughput in batch jobs (ETL, validation, CI)
  - Always aggregate large results for alerting/reporting.
  - Push metrics to Prometheus/Grafana as needed.

Typical real-world goals:
  - Only compute what matters: high accuracy, fast runtimes, minimal monitoring overhead.
  - Results here could be sent to monitoring endpoints or tracked in deployments.

"""
import asyncio
from blazemetrics import BlazeMetricsClient

candidates = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this is a test for async.",
    "Testing parallel execution for blazing fast NLP.",
] * 1000
references = [["The fast brown fox jumps over dog."] for _ in candidates]

client = BlazeMetricsClient()

# Choose ONLY the metrics that are most relevant to your use-case.
# Example: For translation, BLEU is standard. For summarization, ROUGE. For ASR, WER.
# Including less = faster and leaner (as in large production pipelines).
SELECTED_METRICS = ["bleu", "rouge1", "wer"]

print("\n--- Async metrics (ONLY 'rouge1') ---")
async def run_async():
    # Example: Only compute ROUGE-1 in production for summary task APIs.
    fut = client.compute_metrics_async(candidates, references, include=["rouge1"])
    metrics = await fut
    agg = client.aggregate_metrics(metrics)
    print(f"Async Aggregate (rouge1): {agg.get('rouge1_f1', -1):.3f}")
asyncio.run(run_async())

print("\n--- Parallel metrics (ONLY 'bleu' and 'wer') ---")
# Example: Score only BLEU & WER in throughput-optimized parallel batch workflow.
metrics = client.compute_metrics_parallel(
    candidates, references, include=["bleu", "wer"], chunksize=500
)
agg = client.aggregate_metrics(metrics)
print(f"Parallel Aggregate (bleu): {agg.get('bleu', -1):.3f}")
print(f"Parallel Aggregate (wer):  {agg.get('wer', -1):.3f}")

# --- Real-world usage notes:
# - Async example: Use in web services, realtime dashboards, or live cloud APIs.
# - Parallel example: Use for nightly validations, CI/CD batch jobs, or mass batch inferencing.
# - For production, push agg or per-job metrics to Prometheus after this script, as shown in 02_metrics_exporters.py.