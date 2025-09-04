"""
Example: JSON Schema Validation with BlazeMetrics

Demonstrates how to validate and repair structured LLM outputs using JSON schema guardrails.
"""

from blazemetrics.client import BlazeMetricsClient

# Example JSON outputs (some valid, some invalid)
outputs = [
    '{"name": "Alice", "age": 30}',
    '{"name": "Bob", "age": "thirty"}',
    '{"name": "Charlie"}',
    '{"name": 42, "age": 25}',
    'not a json'
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
valid, repaired = client.validate_json(outputs, schema)

for i, (out, v, r) in enumerate(zip(outputs, valid, repaired)):
    print(f"Output {i+1}: {out}")
    print("  Valid:", v)
    print("  Repaired:", r)
    print()