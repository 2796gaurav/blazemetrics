"""
06_compliance_pipeline_gdpr_pii.py - FIXED VERSION

BlazeMetrics Example – Real-Life Compliance, PII, and Schema Analytics Pipeline
-------------------------------------------------------------------------------
This demonstrates:
   - Redacting PII (emails, SSNs, etc) from *arbitrary LLM outputs* (not just static JSON)
   - Enforcing JSON schema validation for structured data
   - Using BlazeMetrics on real OpenAI and HuggingFace outputs (not just hard-coded test data)
   - Analytics for all safety/PII events

How to use:
1. Direct usage: static JSON for PII/schema check (legacy form)
2. LLM integration, OpenAI or HuggingFace (typical prod pipeline):
    - Get LLM output (may be text, JSON, or code)
    - Pass to BlazeMetrics compliance guardrails
    - Enforce policy on output before returning to user

Run this file after:
    pip install openai transformers torch
    # and set your openai key: export OPENAI_API_KEY="sk-..."

----------------- PART 1: Classic static PII/Schema Compliance -----------------
"""
from blazemetrics import BlazeMetricsClient
import json

# Fake rows for test/demo compliance check
texts = [
    '{"name":"Alice Smith", "email":"alice@email.com", "ssn":"123-45-6789"}',
    '{"name":"Bob Brown", "email":"bob@email.com", "ssn":"987-65-4321"}',
]
schema = '{"type":"object","properties":{"name":{"type":"string"},"email":{"type":"string"},"ssn":{"type":"string"}},"required":["name","email","ssn"]}'
client = BlazeMetricsClient(redact_pii=True, enable_analytics=True, json_schema=schema)

results = client.check_safety(texts)
print("-- PII/Schema Compliance Check (Static JSON) --")
for i, r in enumerate(results):
    print(f"Text {i+1}: {texts[i]}")
    print(f"  Redacted: {r.get('redacted')}")
    print(f"  JSON Valid: {r.get('json_valid', [None])[0]}")
print("Analytics:", client.get_analytics_summary())

"""
----------------- PART 2: Real LLM Pipeline Integration ------------------------

Below are two real-life LLM output cases: OpenAI API and HuggingFace transformers.
- In modern chatbot or API pipelines, LLM outputs (OpenAI or HF) can be JSON, text, or code.
- BlazeMetrics compliance checks catch PII/PHI in *any* structure, before returning to end-user.

You may uncomment the section you want to test.
"""

# --- (A) OpenAI API Example: compliance check on LLM outputs ---
print("\n-- OpenAI LLM PII/JSON Compliance Demo --")
import os
import openai
import asyncio

# Initialize OpenAI client properly
openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Use correct chat completions API
result = openai_client.chat.completions.create(
    model="gpt-4o-mini",  # Use valid model name
    messages=[
        {"role": "user", "content": "Return JSON with 'name', 'email', 'ssn' (fake demo data); email=secret@email.com, ssn=654-32-1987."}
    ],
    max_tokens=64,
    temperature=0.1,
)
llm_response = result.choices[0].message.content.strip()
print("LLM raw output:", llm_response)
compliance_results = client.check_safety([llm_response])
print("Redacted:", compliance_results[0].get('redacted'))
print("JSON Valid:", compliance_results[0].get('json_valid', [None])[0])

# --- (B) HuggingFace Transformers Example: compliance check on LLM output ---
print("\n-- HuggingFace LLM PII/JSON Compliance Demo --")
from transformers import pipeline

# Use GPT-2 instead of DistilGPT-2 for better text generation
hf_pipe = pipeline("text-generation", model="gpt2", pad_token_id=50256)
hf_response = hf_pipe(
    "My email is gabi@client.com and my ssn is 321-99-1876. JSON: ", 
    max_new_tokens=40,
    do_sample=True,
    temperature=0.7, # added to prevent timeout
    max_length=50
)[0]["generated_text"]
print("LLM raw output:", hf_response)
# Good: we still catch emails/SSNs in any output structure
compliance_results = client.check_safety([hf_response])
print("Redacted:", compliance_results[0].get('redacted'))
print("JSON Valid:", compliance_results[0].get('json_valid', [None])[0])

"""
=== Summary: ===
- This pattern matches real-world chatbots, API servers, or test pipelines.
- You can wrap all LLM/GenAI completions in compliance/guardrails logic before user return.
- Works equally well for OpenAI, HuggingFace, Google, vLLM, or any provider.
- Analytics are auto-tracked if you enable_analytics=True.
"""