"""
21_inference_api_server_example.py

BlazeMetrics Example – Full Batch + Streaming Inference API Server
-----------------------------------------------------------------
This script demonstrates how to launch a production-grade, FastAPI-powered inference API server for LLM workflows, enhanced with BlazeMetrics for metrics, safety, and guardrails.

--------------------------------------------
WHAT THIS SCRIPT DOES
--------------------------------------------
- Launches a FastAPI API server on port 8095.
- Supports:
  * Synchronous LLM completions (`/completions`)
  * Batch LLM completions (`/completions/batch`)
  * Real-time completions with server-sent events (`/completions/stream`)
  * Reference/baseline metrics computation (`/metrics`)
- Integrates BlazeMetrics for:
  * LLM output metric scoring (BLEU, ROUGE, etc.)
  * Safety & guardrail checks, output filtering, PII redaction
  * Custom real-time business/policy guardrails (user-defined or ML-based)
  * Output logging and tracing for audit/debugging (`/logs` endpoint)
- Ready for extension—add custom endpoints, models, logging, exporters etc. as needed.

--------------------------------------------
HOW TO USE
--------------------------------------------
1. Set your OpenAI API key in the environment: 
   ```bash
   export OPENAI_API_KEY=sk-...
   ```
2. Start the server:
   ```bash
   python examples/21_inference_api_server_example.py
   ```
   - By default, the server runs on port 8095.
3. Make POST requests via `curl` (or Postman/HTTP lib):
   - Single completion:
     ```bash
     curl -X POST http://localhost:8095/completions \
       -H "Content-Type: application/json" \
       -d '{"prompt": "Say hello to BlazeMetrics"}' | jq .
     ```
   - Batch completion:
     ```bash
     curl -X POST http://localhost:8095/completions/batch \
       -H "Content-Type: application/json" \
       -d '{"prompts": ["What is BlazeMetrics?", "Summarize its features"]}'
     ```
   - Compute reference metrics:
     ```bash
     curl -X POST http://localhost:8095/metrics \
       -H "Content-Type: application/json" \
       -d '{"candidates": ["candidate text"], "references": [["reference text"]]}'
     ```
   - See health and logs:
     ```bash
     curl http://localhost:8095/health
     curl http://localhost:8095/logs
     ```
   - Streaming completions via server-sent events are supported at `/completions/stream`.

--------------------------------------------
WHAT TO EXPECT IN API RESPONSES
--------------------------------------------
- `/completions`: Returns a JSON object with your generated output, a unique `trace_id`, LLM evaluation metrics, safety/PII status, guardrail result, and latency (ms).
- `/completions/batch`: Returns lists for each prompt: outputs, metrics, safety, guardrail summaries, trace_ids.
- `/metrics`: Returns BlazeMetrics metric suite for any provided "candidates" and "references", including aggregate scores.
- `/completions/stream`: Returns completion tokens in a stream (with optional streaming guardrail enforcement/chunking).
- `/logs`: Fetches last 100 API calls for auditing/debugging.

--------------------------------------------
NOTES/TIPS
--------------------------------------------
- BlazeMetrics integration is a superset: guardrail/safety checks run live, response includes both LLM metrics and compliance status.
- Default LLM is OpenAI GPT-4o (adjust as needed); you can swap for local LLM or other provider.
- The guardrail logic is a stub (see `guard = RealTimeLLMGuardrail(model=...)`)—extend with your own business/policy model or HuggingFace classifier.
- For full documentation, see BlazeMetrics README and API reference.

--------------------------------------------
FULL CODE BELOW (structure unchanged)
--------------------------------------------
"""
from fastapi import FastAPI, Request, BackgroundTasks
from sse_starlette.sse import EventSourceResponse
from blazemetrics import BlazeMetricsClient
from blazemetrics.llm_guardrails import RealTimeLLMGuardrail
import openai
import os
import time
import uuid
from typing import List
import asyncio

openai.api_key = os.environ.get("OPENAI_API_KEY", "sk-...")

app = FastAPI(title="Async LLM Inference API Server, BlazeMetrics Monitoring")
# BlazeMetricsClient setup:
# - blocklist: terms to block/monitor (e.g., for safety/PII/compliance)
# - redact_pii: automatically redact detected PII in outputs
client = BlazeMetricsClient(blocklist=["hack", "danger", "NSFW"], redact_pii=True)
# RealTimeLLMGuardrail is a live policy classifier/hook over output; see README for custom models
# Here we stub to show basic usage ("bad" => abusive, otherwise safe)
guard = RealTimeLLMGuardrail(model=lambda t: {"label": "abusive" if "bad" in t else "safe", "confidence": 0.9 if "bad" in t else 0.99})

REQUESTS_LOG = []

def log_request(**kwargs):
    kw = dict(kwargs)
    print(f"[TRACE] {kw.get('trace_id')}:", {k: v for k, v in kw.items() if k != 'output'})
    REQUESTS_LOG.append(kw)

@app.post("/completions")
async def completions(request: Request, background_tasks: BackgroundTasks):
    """
    Synchronous LLM completion with metrics, guardrails, logging.
    POST {"prompt": "..."} -> {output, metrics, safety, guardrail, latency_ms}
    """
    data = await request.json()
    prompt = data.get("prompt", "")
    trace_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    result = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=64,
    )

    output = result.choices[0].message.content.strip()
    metrics = client.compute_metrics([output], [[prompt]])
    safety = client.check_safety([output])
    guardrail = guard.validate_full(output)
    t1 = time.perf_counter()

    background_tasks.add_task(
        log_request,
        trace_id=trace_id,
        prompt=prompt,
        output=output,
        metrics=metrics,
        safety=safety,
        guard=guardrail,
        latency_ms=round((t1 - t0) * 1000, 2)
    )

    return {
        "trace_id": trace_id,
        "output": output,
        "metrics": metrics,
        "safety": safety,
        "guardrail": guardrail,
        "latency_ms": round((t1 - t0) * 1000, 2),
    }

@app.post("/completions/batch")
async def completions_batch(request: Request, background_tasks: BackgroundTasks):
    """
    Batch LLM completions and metrics for a list of prompts.
    POST {"prompts": ["prompt 1", ...]} -> results for each
    """
    data = await request.json()
    prompts: List[str] = data.get("prompts", [])
    t0 = time.perf_counter()
    outputs, metrics, safety, guards, trace_ids = [], [], [], [], []

    for prompt in prompts:
        result = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
        )
        out = result.choices[0].message.content.strip()
        met = client.compute_metrics([out], [[prompt]])
        saf = client.check_safety([out])
        guardrail = guard.validate_full(out)
        tid = str(uuid.uuid4())
        outputs.append(out)
        metrics.append(met)
        safety.append(saf)
        guards.append(guardrail)
        trace_ids.append(tid)

        background_tasks.add_task(
            log_request,
            trace_id=tid,
            prompt=prompt,
            output=out,
            metrics=met,
            safety=saf,
            guard=guardrail,
            latency_ms=None
        )

    t1 = time.perf_counter()
    return {
        "trace_ids": trace_ids,
        "outputs": outputs,
        "metrics": metrics,
        "safety": safety,
        "guardrails": guards,
        "latency_ms": round((t1 - t0) * 1000, 2),
    }

@app.post("/metrics")
async def metrics(request: Request):
    """
    Compute BlazeMetrics reference metrics for arbitrary outputs vs references.
    POST {"candidates": [...], "references": [[...], ...]}
    """
    data = await request.json()
    candidates = data.get("candidates", [])
    references = data.get("references", [[] for _ in candidates])
    metrics = client.compute_metrics(candidates, references)
    agg = client.aggregate_metrics(metrics)
    return {"metrics": metrics, "aggregate": agg}

@app.post("/completions/stream")
async def completions_stream(request: Request):
    """
    Streaming completion endpoint (server-sent events; chunked).
    POST {"prompt": "..."} yields tokens/chunks + streaming guardrail status.
    """
    data = await request.json()
    prompt = data.get("prompt", "")

    async def mk_events():
        result = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
            stream=True,
        )
        # For streaming response, we need to handle the stream differently
        full_text = ""
        for chunk in result:
            if chunk.choices[0].delta.content:
                full_text += chunk.choices[0].delta.content
        # Split into tokens for the guardrail
        tokens = full_text.split()
        stream = guard.validate_streaming(iter(tokens), chunk_size=4)
        for i, msg in enumerate(stream):
            event = {"chunk": msg, "index": i}
            yield event

    async def event_generator():
        try:
            async for event in mk_events():
                await asyncio.sleep(0.2)
                yield {"data": event}
        except Exception as e:
            yield {"data": {"error": str(e)}}

    return EventSourceResponse(event_generator())

@app.get("/health")
async def health():
    """
    Simple healthcheck endpoint.
    """
    return {"status": "healthy", "server": "BlazeMetrics LLM API"}

@app.get("/logs")
async def get_logs():
    """
    Fetches the last 100 requests for debug/audit.
    """
    return {"logs": REQUESTS_LOG[-100:]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8095)