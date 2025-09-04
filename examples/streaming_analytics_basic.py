"""
Example: Streaming Analytics with BlazeMetrics

This script demonstrates how to use the StreamingAnalytics engine for real-time metric tracking and anomaly detection.
"""

from blazemetrics.streaming_analytics import StreamingAnalytics, AlertRule

analytics = StreamingAnalytics(
    window_size=10,
    alert_rules=[AlertRule(metric_name="rouge1", threshold=0.5, severity="high", comparison="lt")]
)

# Simulate streaming metric updates
for i in range(20):
    metrics = {"rouge1": 0.4 + 0.05 * (i % 4), "bleu": 0.3 + 0.02 * (i % 3)}
    analytics.add_metrics(metrics)
    print(f"Step {i}: {analytics.get_metric_summary()}")

summary = analytics.get_metric_summary()
print("Aggregated:", summary.get("aggregated_metrics"))
print("Performance stats:", summary.get("performance_stats"))
print("Metric summary:", summary)