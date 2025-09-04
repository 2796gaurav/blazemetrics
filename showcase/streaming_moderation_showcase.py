"""
Showcase: Real-Time Streaming Moderation with BlazeMetrics

Simulates a real-time chat/content moderation pipeline with rolling window analytics and alerts.
"""

import time
from blazemetrics.streaming_analytics import StreamingAnalytics, AlertRule
from blazemetrics.client import BlazeMetricsClient

# Simulated chat stream (some messages are toxic or policy-violating)
chat_stream = [
    "Hello, how are you?",
    "You are an idiot!",
    "Let's meet at 123-456-7890.",
    "This is a normal message.",
    "DROP TABLE users;",
    "My email is alice@example.com.",
    "You are so stupid.",
    "Have a great day!",
    "This contains a bаdword (Cyrillic 'а').",
    "Normal again."
]

blocklist = ["idiot", "stupid", "badword"]

client = BlazeMetricsClient(blocklist=blocklist)

analytics = StreamingAnalytics(
    window_size=5,
    alert_rules=[
        AlertRule(metric_name="blocked", threshold=1, severity="critical", comparison="gt")
    ]
)

print("Starting real-time chat moderation...")
for i, msg in enumerate(chat_stream):
    # Guardrails: blocklist, PII, injection, etc.
    safety = client.check_safety([msg])[0]
    blocked = any(safety.get("blocked", [False]))
    pii_redacted = safety.get("redacted", [msg])[0]
    print(f"Message {i+1}: {msg}")
    print("  Blocked:", blocked)
    print("  PII Redacted:", pii_redacted)
    analytics.add_metrics({"blocked": int(blocked)})
    summary = analytics.get_metric_summary()
    blocked_stats = summary["aggregated_metrics"]["blocked"]
    rolling_count = int(blocked_stats["mean"] * blocked_stats["count"])
    print("  Rolling Blocked Count (last 5):", rolling_count)
    print()
    time.sleep(0.1)