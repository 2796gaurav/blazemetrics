"""
Benchmarking: BlazeMetrics on HuggingFace Pipeline Outputs

Evaluates BlazeMetrics metrics and guardrails on real HuggingFace text-generation outputs.
"""

import time
from blazemetrics.client import BlazeMetricsClient

try:
    from transformers import pipeline
except ImportError:
    print("transformers not installed. Install with 'pip install transformers'")
    exit(0)

hf_pipeline = pipeline("text-generation", model="gpt2")
prompts = [f"Tell me a joke about {topic}" for topic in ["cats", "dogs", "AI", "space", "food"] * 20]
start = time.time()
outputs = [hf_pipeline(prompt, max_new_tokens=20)[0]["generated_text"] for prompt in prompts]
hf_time = time.time() - start
print(f"HuggingFace pipeline generation time: {hf_time:.2f}s")

# Use BlazeMetrics to evaluate outputs
references = [["This is a reference joke."] for _ in outputs]
client = BlazeMetricsClient(blocklist=["idiot", "stupid", "badword"])
start = time.time()
metrics = client.compute_metrics(outputs, references, include=["rouge1", "bleu", "wer"])
guardrails = client.check_safety(outputs)
bm_time = time.time() - start
print(f"BlazeMetrics evaluation time: {bm_time:.2f}s")

print("\n--- Results ---")
print("Sample output:", outputs[0])
print("Sample metrics:", {k: v[0] for k, v in metrics.items()})
print("Sample guardrails:", guardrails[0])