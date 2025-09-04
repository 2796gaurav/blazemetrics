"""
Run All BlazeMetrics Benchmarks

Runs all benchmarking scripts in this directory and summarizes results for easy comparison.
"""

import subprocess
import os

benchmark_scripts = [
    "text_metrics_benchmark.py",
    "rag_semantic_benchmark.py",
    "huggingface_usecase_benchmark.py",
    "json_guardrails_benchmark.py",
    "jsonschema_benchmark.py",
    "profanity_benchmark.py"
]

for script in benchmark_scripts:
    path = os.path.join(os.path.dirname(__file__), script)
    if not os.path.exists(path):
        print(f"Skipping {script} (not found)")
        continue
    print(f"\n=== Running {script} ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    result = subprocess.run(["python3", path], capture_output=True, text=True, env=env)
    print(result.stdout)
    if result.stderr:
        print("Errors/Warnings:", result.stderr)