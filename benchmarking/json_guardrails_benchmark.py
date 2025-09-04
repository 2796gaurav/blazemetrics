"""
Benchmarking: BlazeMetrics JSON Schema Validation & Guardrails

Compares BlazeMetrics' speed and correctness for JSON schema validation, injection detection, and related guardrails on large batches.
"""

import time
from blazemetrics.client import BlazeMetricsClient

# Generate a large batch of JSON outputs (some valid, some invalid)
N = 1000
outputs = [
    '{"name": "Alice", "age": 30}' if i % 3 == 0 else
    '{"name": "Bob", "age": "thirty"}' if i % 3 == 1 else
    'not a json'
    for i in range(N)
]

# Simple schema: name (string), age (integer, optional)
schema = '''
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"}
  },
  "required": ["name"]
}
'''

client = BlazeMetricsClient()

print("Benchmarking BlazeMetrics JSON schema validation...")
start = time.time()
valid, repaired = client.validate_json(outputs, schema)
bm_time = time.time() - start
print(f"BlazeMetrics JSON validation time: {bm_time:.3f}s")
print("Sample valid:", valid[:5])
print("Sample repaired:", repaired[:2])

print("Benchmarking BlazeMetrics injection detection...")
start = time.time()
injection = client.detect_injection(outputs)
inj_time = time.time() - start
print(f"BlazeMetrics injection detection time: {inj_time:.3f}s")
print("Sample injection detection:", injection[:5])