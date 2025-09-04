"""
Benchmarking: BlazeMetrics JSON Schema Validation vs. jsonschema and pydantic

Compares BlazeMetrics' JSON schema validation to jsonschema and pydantic on large batches.
"""

import time
from blazemetrics.client import BlazeMetricsClient

try:
    import jsonschema
except ImportError:
    print("jsonschema not installed. Install with 'pip install jsonschema'")
    jsonschema = None

try:
    from pydantic import BaseModel, ValidationError
except ImportError:
    print("pydantic not installed. Install with 'pip install pydantic'")
    BaseModel = None

import json

# Generate a large batch of JSON outputs (some valid, some invalid)
N = 1000
outputs = [
    '{"name": "Alice", "age": 30}' if i % 3 == 0 else
    '{"name": "Bob", "age": "thirty"}' if i % 3 == 1 else
    'not a json'
    for i in range(N)
]

# Simple schema: name (string), age (integer, optional)
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name"]
}

client = BlazeMetricsClient()

print("Benchmarking BlazeMetrics JSON schema validation...")
start = time.time()
valid, repaired = client.validate_json(outputs, json.dumps(schema))
bm_time = time.time() - start
print(f"BlazeMetrics JSON validation time: {bm_time:.3f}s")
print("Sample valid:", valid[:5])

if jsonschema:
    print("Benchmarking jsonschema...")
    start = time.time()
    js_valid = []
    for s in outputs:
        try:
            js_valid.append(jsonschema.validate(instance=json.loads(s), schema=schema) is None)
        except Exception:
            js_valid.append(False)
    js_time = time.time() - start
    print(f"jsonschema time: {js_time:.3f}s")
    print("Sample jsonschema valid:", js_valid[:5])

if BaseModel:
    print("Benchmarking pydantic...")
    class UserModel(BaseModel):
        name: str
        age: int = None
    start = time.time()
    pd_valid = []
    for s in outputs:
        try:
            UserModel.parse_raw(s)
            pd_valid.append(True)
        except Exception:
            pd_valid.append(False)
    pd_time = time.time() - start
    print(f"pydantic time: {pd_time:.3f}s")
    print("Sample pydantic valid:", pd_valid[:5])