"""
Showcase: Real-Time Streaming Analytics and Alerting with BlazeMetrics

This script demonstrates real-time metric tracking, anomaly detection, and alerting.
"""

from blazemetrics.streaming_analytics import StreamingAnalytics, AlertRule
import random
import time

analytics = StreamingAnalytics(
    window_size=10,
    alert_rules=[
        AlertRule(metric_name="bleu", threshold=0.25, severity="critical", comparison="lt"),
        AlertRule(metric_name="rouge1", threshold=0.5, severity="warning", comparison="lt")
    ]
)

print("Starting real-time metric streaming...")
for i in range(30):
    metrics = {
        "rouge1": random.uniform(0.3, 0.7),
        "bleu": random.uniform(0.1, 0.4)
    }
    analytics.add_metrics(metrics)
    summary = analytics.get_metric_summary()
    print(f"Step {i}: {summary}")
    time.sleep(0.1)  # Simulate real-time data

print("Final performance stats:", analytics.get_performance_stats())