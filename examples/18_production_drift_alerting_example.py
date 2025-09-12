"""
18_production_drift_alerting_example.py

BlazeMetrics Example – Production Drift Alerting, Webhook Export
---------------------------------------------------------------
Example for live production monitoring. Demonstrates:
  - Quality drift detection (with real metric threshold rules)
  - Slack/ServiceNow/Teams webhook integration for on-call alerting
  - Streaming analytics window, alert callback, and notification

Usage:
- Extend the `notify_slack` function for your alerting integration.
- Use for over-time QA, regression, or production AI monitoring/audit.
"""
from blazemetrics import BlazeMetricsClient
from blazemetrics.streaming_analytics import StreamingAnalytics, AlertRule
import requests
import random, time

def notify_slack(alert):
    print(f"[SLACK/TEAMS ALERT HOOK] {alert.severity}: {alert.message}")
    # requests.post("https://hooks.slack.com/services/T00000/B00000/X00000", json={"text": alert.message})

rules = [
    AlertRule(metric_name="bleu", threshold=0.7, comparison="lt", severity="error", message_template="DRIFT: BLEU dropped to {current_value}"),
]
analytics = StreamingAnalytics(window_size=8, alert_rules=rules)
analytics.on_alert = notify_slack
client = BlazeMetricsClient()

print("== Production Drift/Alerting w/ Webhook Export ==")
for i in range(25):
    candidate = "foo bar baz" if i%9==0 else f"Prediction {i+1}"
    reference = [f"Prediction {i+1}"]
    metrics = client.compute_metrics([candidate], [reference])
    agg = client.aggregate_metrics(metrics)
    analytics.add_metrics(agg)
    time.sleep(0.05)
