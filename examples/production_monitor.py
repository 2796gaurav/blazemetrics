"""
Example: Real-Time Production AI Monitoring

Demonstrates how to use ProductionMonitor to track multiple LLMs with
metrics like quality, latency, cost, and safety, including failover and optimization.
"""

from blazemetrics import ProductionMonitor

monitor = ProductionMonitor(
    models=['gpt-4', 'claude-3', 'custom-model'],
    metrics=['quality', 'latency', 'cost', 'safety'],
    alert_thresholds={'quality': 0.8, 'latency': 2.0, 'cost': 0.01},
    a_b_testing=True
)

# Simulate a monitoring loop
tick_count = 0
for metrics in monitor.track_production():
    print("Tick:", tick_count, metrics)

    if metrics.get("quality_drop_detected"):
        monitor.auto_failover_to_backup()

    if metrics.get("cost_spike_detected"):
        monitor.optimize_inference_parameters()

    tick_count += 1
    if tick_count >= 5:  # Run 5 ticks for demonstration
        break