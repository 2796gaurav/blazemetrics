"""
17_streaming_analytics_realtime.py

BlazeMetrics – Real-World Streaming Analytics & Drift Alerting
---------------------------------------------------------------
Production-ready example for LLM/GenAI teams:

- Sliding window, real-time metric aggregation (BLEU, WER, etc.)
- Custom alert rules (Slack, Teams, PagerDuty integration in a one-liner)
- Out-of-the-box Prometheus/StatsD export for dashboards
- Plug directly into your batch or streaming inference/ETL pipeline

Usage:
    - Replace `get_next_predictions()` and `get_next_references()` with your own data/model APIs.
    - Customize alerting to your team's chat/on-call/ops system.
    - Cloud-ready (Prometheus/StatsD metrics export just works).

For Hugging Face or OpenAI users: see further templates at the end!
"""

import time
import requests  # Only needed for real webhooks
from blazemetrics import BlazeMetricsClient, MetricsExporters
from blazemetrics.streaming_analytics import StreamingAnalytics, AlertRule

# Setup metrics exporters: Prometheus, StatsD, etc.
exporter = MetricsExporters(prometheus_gateway="localhost:9091", statsd_addr="localhost:8125")

# Customize this to POST alerts to Slack, Teams, PagerDuty, email, etc.
def notify_slack(alert):
    print(f"[PRODUCTION ALERT] {alert.severity}: {alert.message}")
    # To integrate with Slack:
    # requests.post("https://hooks.slack.com/services/XXX/YYY", json={"text": alert.message})

# Set alert rules: add as many as you need for your org
rules = [
    AlertRule(
        metric_name="bleu",
        threshold=0.65,
        comparison="lt",
        severity="error",
        message_template="BLEU score dropped to {current_value:.3f}! Investigate."
    ),
    AlertRule(
        metric_name="wer",
        threshold=0.35,
        comparison="gt",
        severity="critical",
        message_template="WER spiked to {current_value:.3f}! Regression suspected."
    )
]

# Initialize analytics pipeline: window size controls trend smoothing
analytics = StreamingAnalytics(window_size=10, alert_rules=rules)
analytics.on_alert = notify_slack  # Connect alerts to your notification system

client = BlazeMetricsClient()

# Example: replace below with your real model/data pipeline
def get_next_predictions():
    """Yield batches of predictions/results from your model (can be async if desired)"""
    for i in range(200):
        # Simulate occasional regression
        if i % 13 == 0:
            yield "low quality placeholder output"
        else:
            yield f"Expected output sentence {i}"

def get_next_references():
    """Yield ground truth references (must align with predictions above)"""
    for i in range(200):
        yield [f"Expected output sentence {i}"]  # List of refs per sample

print("== Real-time BlazeMetrics Streaming Analytics ==")
windowed_preds = []
windowed_refs = []
for step, (pred, ref) in enumerate(zip(get_next_predictions(), get_next_references())):
    windowed_preds.append(pred)
    windowed_refs.append(ref)
    # Feed to metrics only once window fills for fair stats
    if len(windowed_preds) < analytics.window_size:
        continue
    # Only keep rolling window
    windowed_preds = windowed_preds[-analytics.window_size:]
    windowed_refs = windowed_refs[-analytics.window_size:]
    # Compute evaluation metrics
    metrics = client.compute_metrics(windowed_preds, windowed_refs)
    agg = client.aggregate_metrics(metrics)
    analytics.add_metrics(agg)  # Triggers alert if any rules fire
    exporter.export(agg, labels={"step": str(step)})  # Cloud monitoring/dashboards

    if step % analytics.window_size == 0:
        print(f"Step {step}: Window analytics:", analytics.get_metric_summary())
    # Optional: sleep (or remove in async/production)
    time.sleep(0.05)

print("== Streaming analytics completed. Check your alert channels and dashboard exports. ==")


""" Huggingface """

# hf_streaming_analytics.py
from transformers import pipeline
import sacrebleu
import requests

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
prometheus_url = "http://localhost:9091/metrics"

slack_webhook = "https://hooks.slack.com/services/XXX/YYY/ZZZ"

def get_payloads_and_refs():
    for i in range(100):
        payload = f"Document to summarize {i}"
        ref = f"Expected summary {i}"
        yield payload, ref

window_size = 10
windowed_pred = []
windowed_ref = []
for i, (text, ref) in enumerate(get_payloads_and_refs()):
    pred = summarizer(text)[0]['summary_text']
    windowed_pred.append(pred)
    windowed_ref.append(ref)
    if len(windowed_pred) < window_size:
        continue
    windowed_pred = windowed_pred[-window_size:]
    windowed_ref = windowed_ref[-window_size:]
    bleu = sacrebleu.corpus_bleu(windowed_pred, [windowed_ref]).score
    # Example Prometheus export and Slack alert
    # (Pushgateway or custom exporter needed for Prometheus)
    if bleu < 25:
        requests.post(slack_webhook, json={"text": f"HF BLEU Alert: {bleu:.1f}"})
    print(f"Window {i//window_size}: BLEU={bleu:.2f}")


"""" openai """

# openai_streaming_analytics.py
import openai, sacrebleu, requests

openai.api_key = "sk-..."

def gen_and_eval(prompts, refs, window_size=10):
    windowed_pred, windowed_ref = [], []
    slack_webhook = "https://hooks.slack.com/services/XXX/YYY/ZZZ"
    for i, (prompt, ref) in enumerate(zip(prompts, refs)):
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        pred = resp.choices[0].message.content.strip()
        windowed_pred.append(pred)
        windowed_ref.append(ref)
        if len(windowed_pred) < window_size:
            continue
        windowed_pred = windowed_pred[-window_size:]
        windowed_ref = windowed_ref[-window_size:]
        bleu = sacrebleu.corpus_bleu(windowed_pred, [windowed_ref]).score
        if bleu < 25:
            requests.post(slack_webhook, json={"text": f"OpenAI BLEU Alert: {bleu:.1f}"})
        print(f"Window {i//window_size}: BLEU={bleu:.2f}")

prompts = [f"Summarize input {i}" for i in range(100)]
refs = [f"Expected summary {i}" for i in range(100)]
gen_and_eval(prompts, refs)